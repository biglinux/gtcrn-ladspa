//! GTCRN LADSPA Plugin implementation.
//!
//! Processes audio synchronously within run() for compatibility with
//! both real-time hosts (PipeWire/PulseAudio) and offline hosts (ffmpeg).
//!
//! ## Architecture
//!
//! STFT size adapts to the host sample rate so the first 257 bins always
//! cover 0–8 kHz (matching the 16 kHz model). No sample-rate conversion.
//!
//! ```text
//! run() callback (synchronous)
//! ┌──────────────────────────────────────────────┐
//! │ input → STFT(N) → scale → Model → unscale   │
//! │         → iSTFT(N) → out                     │
//! └──────────────────────────────────────────────┘
//! ```

use crate::biquad::Biquad;
use crate::model::{GtcrnModel, ModelType, NUM_FREQ_BINS};
use crate::stft::StftProcessor;
use crate::{PORT_ENABLE, PORT_INPUT, PORT_MODEL, PORT_OUTPUT, PORT_STRENGTH};
use ladspa::{Plugin, PluginDescriptor, PortConnection};

// =============================================================================
// Constants
// =============================================================================

/// Highpass pre-filter cutoff frequency in Hz.
/// Removes sub-audible content that confuses the GTCRN model.
const HP_CUTOFF_HZ: f32 = 80.0;

// =============================================================================
// Plugin Implementation
// =============================================================================

/// How often (in `run()` calls) to re-read the external control file.
/// The file lives on tmpfs so reads are just page-cache lookups (~1 μs).
const EXTERNAL_POLL_INTERVAL: u32 = 1;

/// Well-known path inside `$XDG_RUNTIME_DIR` where the media player
/// writes live control values (strength, model_type) as two little-endian
/// f32 values (8 bytes total).
const CONTROL_FILENAME: &str = "gtcrn-ladspa-controls";

// =============================================================================
// External Controls (shared-file mechanism)
// =============================================================================

/// Reads live control values from a small file on tmpfs, avoiding the need
/// to recreate LADSPA instances when the user adjusts the slider.
struct ExternalControls {
    path: std::path::PathBuf,
    strength: f32,
    model_type: f32,
    available: bool,
    counter: u32,
}

impl ExternalControls {
    fn new() -> Self {
        let path = std::env::var("XDG_RUNTIME_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
            .join(CONTROL_FILENAME);

        let mut ctrl = Self {
            path,
            strength: -1.0,
            model_type: -1.0,
            available: false,
            counter: EXTERNAL_POLL_INTERVAL, // trigger immediate read
        };
        ctrl.poll();
        ctrl
    }

    /// Re-read the control file if enough calls have elapsed.
    fn poll(&mut self) {
        self.counter += 1;
        if self.counter < EXTERNAL_POLL_INTERVAL {
            return;
        }
        self.counter = 0;
        if let Ok(data) = std::fs::read(&self.path) {
            if data.len() >= 8 {
                self.strength = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                self.model_type = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                self.available = true;
            }
        }
    }
}

// =============================================================================
// Plugin
// =============================================================================

/// GTCRN LADSPA plugin instance.
///
/// All processing happens synchronously within `run()`. STFT size adapts
/// to the host sample rate so no sample-rate conversion is needed.
///
/// The ONNX model is lazily initialized on first `run()` call.
pub struct GtcrnPlugin {
    model: Option<GtcrnModel>,
    stft: StftProcessor,

    /// Computed once from stft at init.
    nfft: usize,
    hop_size: usize,

    /// Highpass pre-filter (removes sub-80Hz content)
    hp_filter: Biquad,

    /// Sliding analysis window (last nfft samples)
    window: Vec<f32>,

    /// Input accumulator (feeds STFT in hop_size chunks)
    input_accum: Vec<f32>,
    input_accum_len: usize,

    /// Output accumulator (synthesised samples waiting to be sent)
    output_accum: Vec<f32>,
    output_accum_len: usize,

    /// Spectrum scratch buffer
    spectrum_buffer: [(f32, f32); NUM_FREQ_BINS],

    /// External control file for live parameter updates
    ext_controls: ExternalControls,
}

impl GtcrnPlugin {
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(_descriptor: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
        let stft = StftProcessor::new(sample_rate as u32);
        let nfft = stft.nfft();
        let hop_size = stft.hop_size();
        let hp_filter = Biquad::highpass(HP_CUTOFF_HZ, sample_rate as f32);

        Box::new(Self {
            model: None,
            stft,
            nfft,
            hop_size,
            hp_filter,
            window: vec![0.0; nfft],
            input_accum: vec![0.0; hop_size * 4],
            input_accum_len: 0,
            output_accum: vec![0.0; hop_size * 8],
            output_accum_len: 0,
            spectrum_buffer: [(0.0, 0.0); NUM_FREQ_BINS],
            ext_controls: ExternalControls::new(),
        })
    }

    /// Process complete STFT frames: analyze → model → synthesize.
    fn process_frames(&mut self, strength: f32) {
        let hop = self.hop_size;
        let max_frames = self.input_accum_len / hop;
        self.ensure_output_accum(max_frames * hop);

        while self.input_accum_len >= hop {
            // Shift window and add new samples
            self.window.copy_within(hop.., 0);
            self.window[self.nfft - hop..].copy_from_slice(&self.input_accum[..hop]);
            self.input_accum.copy_within(hop..self.input_accum_len, 0);
            self.input_accum_len -= hop;

            // STFT analysis (already scaled for model)
            let spectrum = self.stft.analyze(&self.window);
            self.spectrum_buffer.copy_from_slice(spectrum);

            // Model inference + strength blending
            let final_spectrum = if strength == 0.0 {
                self.spectrum_buffer
            } else {
                let enhanced = self
                    .model
                    .as_mut()
                    .expect("model must be initialized before processing")
                    .process_frame(&self.spectrum_buffer)
                    .unwrap_or_else(|e| {
                        eprintln!("GTCRN model error: {e}");
                        self.spectrum_buffer
                    });

                if strength < 1.0 {
                    let mut blended = self.spectrum_buffer;
                    for i in 0..NUM_FREQ_BINS {
                        blended[i].0 =
                            self.spectrum_buffer[i].0 * (1.0 - strength) + enhanced[i].0 * strength;
                        blended[i].1 =
                            self.spectrum_buffer[i].1 * (1.0 - strength) + enhanced[i].1 * strength;
                    }
                    blended
                } else {
                    enhanced
                }
            };

            // iSTFT synthesis (un-scales internally)
            let processed = self.stft.synthesize(&final_spectrum);
            let proc_len = processed.len();

            self.output_accum[self.output_accum_len..self.output_accum_len + proc_len]
                .copy_from_slice(processed);
            self.output_accum_len += proc_len;
        }
    }

    #[inline]
    fn ensure_output_accum(&mut self, extra: usize) {
        let needed = self.output_accum_len + extra;
        if needed > self.output_accum.len() {
            self.output_accum.resize(needed + self.hop_size, 0.0);
        }
    }
}

impl Plugin for GtcrnPlugin {
    fn activate(&mut self) {}

    fn deactivate(&mut self) {}

    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input = ports[PORT_INPUT].unwrap_audio();
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut();
        let enable_control = *ports[PORT_ENABLE].unwrap_control();
        let strength_control = *ports[PORT_STRENGTH].unwrap_control();
        let model_control = *ports[PORT_MODEL].unwrap_control();

        // Poll external control file for live parameter updates
        self.ext_controls.poll();

        // External file overrides LADSPA port values when available
        let (strength_val, model_val) = if self.ext_controls.available {
            (self.ext_controls.strength, self.ext_controls.model_type)
        } else {
            (strength_control, model_control)
        };

        // Bypass mode: pass through input directly
        if enable_control < 0.5 || strength_val <= 0.0 {
            output[..sample_count].copy_from_slice(&input[..sample_count]);
            return;
        }

        // Update model type if changed (or create on first call)
        let requested = ModelType::from_control(model_val);
        match &self.model {
            None => {
                self.model = Some(GtcrnModel::new(requested));
            }
            Some(m) if m.model_type() != requested => {
                self.model
                    .as_mut()
                    .expect("model must be initialized")
                    .set_model_type(requested);
            }
            _ => {}
        }

        let strength = strength_val.clamp(0.0, 1.0);

        // Append new input to accumulator (after HP pre-filter)
        let needed = self.input_accum_len + sample_count;
        if needed > self.input_accum.len() {
            self.input_accum.resize(needed + self.hop_size, 0.0);
        }
        self.input_accum[self.input_accum_len..self.input_accum_len + sample_count]
            .copy_from_slice(&input[..sample_count]);
        // Apply highpass in-place on the newly appended samples
        self.hp_filter.process(
            &mut self.input_accum[self.input_accum_len..self.input_accum_len + sample_count],
        );
        self.input_accum_len += sample_count;

        // Process pipeline: STFT → model → iSTFT
        self.process_frames(strength);

        // Copy available output to host buffer
        let available = self.output_accum_len.min(sample_count);
        output[..available].copy_from_slice(&self.output_accum[..available]);

        // Fill remainder with silence (initial latency frames only)
        if available < sample_count {
            output[available..sample_count].fill(0.0);
        }

        // Shift output accumulator
        if available > 0 {
            self.output_accum
                .copy_within(available..self.output_accum_len, 0);
            self.output_accum_len -= available;
        }
    }
}
