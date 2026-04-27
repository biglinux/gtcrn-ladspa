//! GTCRN LADSPA Plugin implementation.
//!
//! Processes audio synchronously inside `run()` for compatibility with
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
//! │ input → HP → STFT → Model → blend → iSTFT    │
//! │       → output → gate-pass                   │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! Heavy lifting is delegated to dedicated modules:
//! - [`crate::external_controls`] — live parameter polling from tmpfs.
//! - [`crate::vad`] — voice-activity detection helpers.
//! - [`crate::gate_pass`] — sidechain noise gate post-stage.

use std::collections::VecDeque;

use realfft::num_complex::Complex;

use crate::biquad::Biquad;
use crate::external_controls::ExternalControls;
use crate::gate_pass::{apply_gate_pass, GateState};
use crate::model::{GtcrnModel, ModelType, NUM_FREQ_BINS};
use crate::stft::StftProcessor;
use crate::vad::{
    detect_onset, dns3_energy_ratio_vad, effective_strength, spectrum_energy, step_vad_gate,
    update_noise_floor, vctk_input_only_vad,
};
use crate::{
    PORT_ENABLE, PORT_INPUT, PORT_LOOKAHEAD_MS, PORT_MODEL, PORT_MODEL_BLEND, PORT_OUTPUT,
    PORT_SPEECH_STRENGTH, PORT_STRENGTH, PORT_VOICE_RECOVERY,
};
use ladspa::{Plugin, PluginDescriptor, PortConnection};

// =============================================================================
// Constants
// =============================================================================

/// Highpass pre-filter cutoff (Hz). Removes sub-audible content that
/// confuses the GTCRN model.
const HP_CUTOFF_HZ: f32 = 80.0;

/// Frames with `model_blend = false` before the secondary model unloads.
/// At ~62 fps (48 kHz, hop=768) this is ≈1.6 s.
const DUAL_MODEL_UNLOAD_FRAMES: usize = 100;

/// Maximum host audio block we pre-allocate buffers for. 32k samples is
/// ~680 ms at 48 kHz — well above any realistic LADSPA host block. Hosts
/// that exceed this still work via `Vec::resize`, but the common path
/// stays allocation-free.
const MAX_HOST_BLOCK: usize = 32 * 1024;

/// How many spectrum buffers to keep in the lookahead pool. Sized for the
/// largest practical lookahead (~200 ms / hop) plus headroom — beyond
/// this the pool falls back to allocating a fresh `Vec` on the audio
/// thread, which is what we want to avoid.
const SPECTRUM_POOL_CAPACITY: usize = 32;

/// Per-frame normalization constants. The model was trained on typical
/// speech levels; quiet input has magnitudes too small for it to
/// distinguish from noise, causing over-suppression.
const NORM_TARGET_RMS: f32 = 0.5;
const NORM_MIN_RMS: f32 = 0.02;
const NORM_MAX_GAIN: f32 = 3.0;
const NORM_EMA: f32 = 0.15;

// =============================================================================
// Process Configuration
// =============================================================================

/// All resolved per-block parameters used inside [`GtcrnPlugin::process_frames`].
struct ProcessConfig {
    noise_strength: f32,
    speech_strength: f32,
    model_blend: bool,
    /// Voice recovery level (0.0 = cut all >8 kHz, 1.0 = full original).
    voice_recovery: f32,
}

/// Resolved block-level controls (LADSPA port values mixed with live
/// values from [`ExternalControls`]).
struct RuntimeControls {
    enable: f32,
    strength: f32,
    model: f32,
    speech_strength: f32,
    lookahead_ms: f32,
    model_blend: bool,
    voice_recovery: f32,
}

// =============================================================================
// Buffered frame (lookahead)
// =============================================================================

/// One STFT frame held in the lookahead queue while we decide its blend.
struct BufferedFrame {
    original: [(f32, f32); NUM_FREQ_BINS],
    enhanced: [(f32, f32); NUM_FREQ_BINS],
    /// Secondary-model output (VCTK during speech) for dual-model blending.
    enhanced_alt: [(f32, f32); NUM_FREQ_BINS],
    dual_model: bool,
    is_speech: bool,
    /// Original full spectrum (all `n_bins`) used for HF reconstruction.
    original_full_spectrum: Vec<Complex<f32>>,
}

// =============================================================================
// Plugin
// =============================================================================

/// GTCRN LADSPA plugin instance with VAD-driven dual-strength and lookahead.
pub struct GtcrnPlugin {
    model: Option<GtcrnModel>,
    second_model: Option<GtcrnModel>,
    stft: StftProcessor,

    sample_rate: f32,
    nfft: usize,
    hop_size: usize,

    hp_filter: Biquad,

    /// Sliding analysis window (last `nfft` samples).
    window: Vec<f32>,
    /// Input accumulator (feeds STFT in `hop_size` chunks).
    input_accum: Vec<f32>,
    input_accum_len: usize,
    /// Output accumulator (synthesised samples awaiting emission).
    output_accum: Vec<f32>,
    output_accum_len: usize,

    spectrum_buffer: [(f32, f32); NUM_FREQ_BINS],

    ext_controls: ExternalControls,

    // VAD + lookahead
    vad_gate: f32,
    vad_energy_smooth: f32,
    lookahead_buf: VecDeque<BufferedFrame>,
    lookahead_frames: usize,

    // Feature state
    model_blend_zero_count: usize,
    /// Running input RMS for boosting quiet audio before model inference.
    input_norm_rms: f32,
    /// EMA of input energy during silence — feeds onset detection.
    input_noise_floor: f32,

    /// Pre-allocated pool of spectrum buffers for lookahead frames.
    /// Sized at construction so the audio thread stays allocation-free.
    spectrum_pool: Vec<Vec<Complex<f32>>>,

    gate: GateState,
}

impl GtcrnPlugin {
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(_descriptor: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
        let sr = sample_rate as f32;
        let stft = StftProcessor::new(sample_rate as u32);
        let nfft = stft.nfft();
        let hop_size = stft.hop_size();
        let hp_filter = Biquad::highpass(HP_CUTOFF_HZ, sr);
        let n_bins = nfft / 2 + 1;
        let spectrum_pool: Vec<Vec<Complex<f32>>> = (0..SPECTRUM_POOL_CAPACITY)
            .map(|_| vec![Complex::new(0.0, 0.0); n_bins])
            .collect();
        let input_accum_cap = MAX_HOST_BLOCK + hop_size * 4;
        let output_accum_cap = MAX_HOST_BLOCK + hop_size * 8;

        Box::new(Self {
            model: None,
            second_model: None,
            stft,
            sample_rate: sr,
            nfft,
            hop_size,
            hp_filter,
            window: vec![0.0; nfft],
            input_accum: vec![0.0; input_accum_cap],
            input_accum_len: 0,
            output_accum: vec![0.0; output_accum_cap],
            output_accum_len: 0,
            spectrum_buffer: [(0.0, 0.0); NUM_FREQ_BINS],
            ext_controls: ExternalControls::new(),
            vad_gate: 0.0,
            vad_energy_smooth: 0.0,
            // Sized for the largest practical lookahead (~200 ms / hop ≈ 13 frames).
            // Avoids growing inside the audio callback when the user pushes
            // lookahead to its max.
            lookahead_buf: VecDeque::with_capacity(SPECTRUM_POOL_CAPACITY),
            lookahead_frames: 0,
            model_blend_zero_count: 0,
            input_norm_rms: 0.0,
            input_noise_floor: 0.0,
            spectrum_pool,
            gate: GateState::new(sr, MAX_HOST_BLOCK),
        })
    }

    /// Convert milliseconds into STFT frames.
    fn ms_to_frames(&self, ms: f32) -> usize {
        if ms <= 0.0 {
            return 0;
        }
        let frame_ms = (self.hop_size as f32) / self.sample_rate * 1000.0;
        (ms / frame_ms).round() as usize
    }

    // ---------------------------------------------------------------------
    // run() helpers
    // ---------------------------------------------------------------------

    /// Resolve runtime controls — port values overridden by the live
    /// external-controls file when it's available.
    fn resolve_controls<'a>(&mut self, ports: &[&'a PortConnection<'a>]) -> RuntimeControls {
        self.ext_controls.poll();
        let ec_avail = self.ext_controls.available;

        RuntimeControls {
            enable: *ports[PORT_ENABLE].unwrap_control(),
            strength: pick_ext(
                ec_avail,
                self.ext_controls.strength,
                *ports[PORT_STRENGTH].unwrap_control(),
            ),
            model: pick_ext(
                ec_avail,
                self.ext_controls.model_type,
                *ports[PORT_MODEL].unwrap_control(),
            ),
            speech_strength: pick_ext_nonneg(
                ec_avail,
                self.ext_controls.speech_strength,
                *ports[PORT_SPEECH_STRENGTH].unwrap_control(),
            ),
            lookahead_ms: pick_ext_nonneg(
                ec_avail,
                self.ext_controls.lookahead_ms,
                *ports[PORT_LOOKAHEAD_MS].unwrap_control(),
            ),
            model_blend: pick_ext(
                ec_avail,
                self.ext_controls.model_blend,
                *ports[PORT_MODEL_BLEND].unwrap_control(),
            ) >= 0.5,
            voice_recovery: pick_ext_nonneg(
                ec_avail,
                self.ext_controls.voice_recovery,
                *ports[PORT_VOICE_RECOVERY].unwrap_control(),
            ),
        }
    }

    /// Ensure the primary model is loaded and matches the requested type.
    /// On ONNX session failure the model stays `None`; subsequent
    /// `process_frames` calls degrade gracefully (dry-bypass through the
    /// scaled spectrum).
    fn update_primary_model(&mut self, requested: ModelType) {
        match &self.model {
            None => self.model = GtcrnModel::new(requested),
            Some(m) if m.model_type() != requested => {
                if let Some(m) = self.model.as_mut() {
                    m.set_model_type(requested);
                }
            }
            _ => {}
        }
    }

    /// Manage the secondary model lifecycle for dual-model blending.
    fn update_secondary_model(&mut self, model_blend: bool) {
        if model_blend {
            self.model_blend_zero_count = 0;
            let primary_type = self
                .model
                .as_ref()
                .map_or(ModelType::Dns3, GtcrnModel::model_type);
            let secondary_type = match primary_type {
                ModelType::Dns3 => ModelType::Vctk,
                ModelType::Vctk => ModelType::Dns3,
            };
            if self.second_model.is_none() {
                self.second_model = GtcrnModel::new(secondary_type);
            }
            return;
        }
        self.model_blend_zero_count += 1;
        if self.model_blend_zero_count > DUAL_MODEL_UNLOAD_FRAMES && self.second_model.is_some() {
            self.second_model = None;
        }
    }

    /// Append `input` to the accumulator and run the HP pre-filter on the
    /// freshly-appended slice.
    fn feed_input(&mut self, input: &[f32], sample_count: usize) {
        let needed = self.input_accum_len + sample_count;
        if needed > self.input_accum.len() {
            self.input_accum.resize(needed + self.hop_size, 0.0);
        }
        self.input_accum[self.input_accum_len..self.input_accum_len + sample_count]
            .copy_from_slice(&input[..sample_count]);
        self.hp_filter.process(
            &mut self.input_accum[self.input_accum_len..self.input_accum_len + sample_count],
        );
        self.input_accum_len += sample_count;
    }

    /// Drain up to `sample_count` samples from `output_accum` into the host
    /// buffer; pad with zeros and shift the residual down.
    fn emit_output(&mut self, output: &mut [f32], sample_count: usize) {
        let available = self.output_accum_len.min(sample_count);
        output[..available].copy_from_slice(&self.output_accum[..available]);
        if available < sample_count {
            output[available..sample_count].fill(0.0);
        }
        if available > 0 {
            self.output_accum
                .copy_within(available..self.output_accum_len, 0);
            self.output_accum_len -= available;
        }
    }

    // ---------------------------------------------------------------------
    // process_frames helpers
    // ---------------------------------------------------------------------

    fn ensure_output_accum(&mut self, extra: usize) {
        let needed = self.output_accum_len + extra;
        if needed > self.output_accum.len() {
            self.output_accum.resize(needed + self.hop_size, 0.0);
        }
    }

    /// Slide the analysis window forward by `hop_size`, refilling from
    /// `input_accum`. Caller must ensure `input_accum_len >= hop_size`.
    fn shift_window(&mut self) {
        let hop = self.hop_size;
        self.window.copy_within(hop.., 0);
        self.window[self.nfft - hop..].copy_from_slice(&self.input_accum[..hop]);
        self.input_accum.copy_within(hop..self.input_accum_len, 0);
        self.input_accum_len -= hop;
    }

    /// Compute the input-normalisation gain for this frame and update the
    /// running RMS estimate. Returns `1.0` when no boost should apply.
    fn compute_norm_gain(&mut self) -> f32 {
        let mag_sq_sum: f32 = self
            .spectrum_buffer
            .iter()
            .map(|&(re, im)| re * re + im * im)
            .sum();
        let frame_rms = (mag_sq_sum / NUM_FREQ_BINS as f32).sqrt();
        self.input_norm_rms = if self.input_norm_rms < 1e-6 {
            frame_rms
        } else {
            self.input_norm_rms * (1.0 - NORM_EMA) + frame_rms * NORM_EMA
        };
        if self.input_norm_rms > NORM_MIN_RMS && self.input_norm_rms < NORM_TARGET_RMS {
            (NORM_TARGET_RMS / self.input_norm_rms).min(NORM_MAX_GAIN)
        } else {
            1.0
        }
    }

    /// Run the primary model on `model_input`. On error the original
    /// scaled spectrum is returned — the audio path stays alive.
    fn run_primary(
        &mut self,
        model_input: &[(f32, f32); NUM_FREQ_BINS],
    ) -> [(f32, f32); NUM_FREQ_BINS] {
        match self.model.as_mut() {
            Some(m) => m.process_frame(model_input).unwrap_or(self.spectrum_buffer),
            None => self.spectrum_buffer,
        }
    }

    /// Run the secondary model when dual-model blending is on; ensures
    /// `enhanced` ends up as the DNS3 output and `enhanced_alt` as VCTK.
    fn run_secondary(
        &mut self,
        model_input: &[(f32, f32); NUM_FREQ_BINS],
        enhanced: &mut [(f32, f32); NUM_FREQ_BINS],
        norm_gain: f32,
    ) -> [(f32, f32); NUM_FREQ_BINS] {
        let mut enhanced_alt = [(0.0_f32, 0.0_f32); NUM_FREQ_BINS];
        let Some(second) = self.second_model.as_mut() else {
            return enhanced_alt;
        };
        enhanced_alt = second
            .process_frame(model_input)
            .unwrap_or(self.spectrum_buffer);
        if norm_gain > 1.01 {
            apply_gain(&mut enhanced_alt, 1.0 / norm_gain);
        }
        let primary_is_dns3 = self
            .model
            .as_ref()
            .is_none_or(|m| m.model_type() == ModelType::Dns3);
        if !primary_is_dns3 {
            std::mem::swap(enhanced, &mut enhanced_alt);
        }
        enhanced_alt
    }

    /// Per-frame VAD decision combining DNS3 (energy ratio) or VCTK
    /// (input vs. noise floor) flavours, depending on which model is in
    /// `enhanced`.
    fn frame_vad(
        &mut self,
        input_energy: f32,
        enhanced: &[(f32, f32); NUM_FREQ_BINS],
        dual_model: bool,
    ) -> bool {
        let primary_is_dns3 = dual_model
            || self
                .model
                .as_ref()
                .is_none_or(|m| m.model_type() == ModelType::Dns3);
        if primary_is_dns3 {
            let output_energy = spectrum_energy(enhanced);
            let (is_speech, ema) =
                dns3_energy_ratio_vad(input_energy, output_energy, self.vad_energy_smooth);
            self.vad_energy_smooth = ema;
            is_speech
        } else {
            vctk_input_only_vad(input_energy, self.input_noise_floor)
        }
    }

    /// Borrow a spectrum-shaped buffer from the pool; allocates only when
    /// the pool is exhausted (sizing should keep this off the audio thread).
    fn pop_pool_buffer(&mut self, len: usize) -> Vec<Complex<f32>> {
        let mut buf = self
            .spectrum_pool
            .pop()
            .unwrap_or_else(|| vec![Complex::new(0.0, 0.0); len]);
        buf.resize(len, Complex::new(0.0, 0.0));
        buf
    }

    /// Mix the dual-model outputs with the VAD-driven crossfade.
    /// `frame.enhanced` is DNS3 (silence side), `enhanced_alt` is VCTK
    /// (speech side).
    fn mix_dual_model(&self, frame: &BufferedFrame) -> [(f32, f32); NUM_FREQ_BINS] {
        if !frame.dual_model {
            return frame.enhanced;
        }
        let speech_mix = ((self.vad_gate - 0.2) / 0.5).clamp(0.0, 1.0);
        if speech_mix <= 0.0 {
            return frame.enhanced;
        }
        if speech_mix >= 1.0 {
            return frame.enhanced_alt;
        }
        let mut mixed = frame.enhanced;
        let inv = 1.0 - speech_mix;
        for (m, (e, a)) in mixed
            .iter_mut()
            .zip(frame.enhanced.iter().zip(frame.enhanced_alt.iter()))
        {
            m.0 = e.0 * inv + a.0 * speech_mix;
            m.1 = e.1 * inv + a.1 * speech_mix;
        }
        mixed
    }

    /// Process one analysis frame: STFT → models → push to lookahead queue.
    fn analyse_one_frame(&mut self, cfg: &ProcessConfig) {
        self.shift_window();
        let spectrum = self.stft.analyze(&self.window);
        self.spectrum_buffer.copy_from_slice(spectrum);

        let mut model_input = self.spectrum_buffer;
        let norm_gain = self.compute_norm_gain();
        if norm_gain > 1.01 {
            apply_gain(&mut model_input, norm_gain);
        }

        let mut enhanced = self.run_primary(&model_input);
        if norm_gain > 1.01 {
            apply_gain(&mut enhanced, 1.0 / norm_gain);
        }

        let dual_model = cfg.model_blend;
        let enhanced_alt = if dual_model {
            self.run_secondary(&model_input, &mut enhanced, norm_gain)
        } else {
            [(0.0_f32, 0.0_f32); NUM_FREQ_BINS]
        };

        let input_energy = spectrum_energy(&self.spectrum_buffer);
        let is_speech = self.frame_vad(input_energy, &enhanced, dual_model);

        let orig_len = self.stft.original_spectrum().len();
        let mut orig_spectrum = self.pop_pool_buffer(orig_len);
        orig_spectrum.copy_from_slice(self.stft.original_spectrum());

        self.lookahead_buf.push_back(BufferedFrame {
            original: self.spectrum_buffer,
            enhanced,
            enhanced_alt,
            dual_model,
            is_speech,
            original_full_spectrum: orig_spectrum,
        });
    }

    /// Pop the oldest lookahead frame, run VAD-gate update + onset
    /// detector + dual-model mix + iSTFT, append to `output_accum`.
    fn emit_one_frame(&mut self, cfg: &ProcessConfig) {
        let Some(frame) = self.lookahead_buf.pop_front() else {
            return;
        };
        let any_future_speech = self.lookahead_buf.iter().any(|f| f.is_speech);
        let input_energy = spectrum_energy(&frame.original);

        update_noise_floor(
            &mut self.input_noise_floor,
            input_energy,
            self.vad_gate,
            frame.is_speech,
        );
        let onset = detect_onset(input_energy, self.input_noise_floor);
        step_vad_gate(
            &mut self.vad_gate,
            &mut self.vad_energy_smooth,
            frame.is_speech,
            any_future_speech,
            onset,
        );

        let strength = effective_strength(cfg.noise_strength, cfg.speech_strength, self.vad_gate);
        let chosen_enhanced = self.mix_dual_model(&frame);
        let final_spectrum = blend_strength(&frame.original, &chosen_enhanced, strength);

        let synth = self.stft.synthesize(
            &final_spectrum,
            &frame.original_full_spectrum,
            self.vad_gate,
            cfg.voice_recovery,
        );
        let proc_len = synth.len();
        let processed_start = self.output_accum_len;
        self.output_accum[processed_start..processed_start + proc_len].copy_from_slice(synth);
        self.output_accum_len += proc_len;

        // Cap pool growth so a brief lookahead spike doesn't permanently
        // inflate memory. Drop the buffer past the cap.
        if self.spectrum_pool.len() < SPECTRUM_POOL_CAPACITY {
            self.spectrum_pool.push(frame.original_full_spectrum);
        }
    }

    /// Drive the STFT loop: consume `input_accum` in `hop_size` chunks,
    /// run analysis + model + emission for each whole frame.
    fn process_frames(&mut self, cfg: &ProcessConfig) {
        let hop = self.hop_size;
        let max_frames = self.input_accum_len / hop;
        self.ensure_output_accum(max_frames * hop);

        while self.input_accum_len >= hop {
            self.analyse_one_frame(cfg);
            while self.lookahead_buf.len() > self.lookahead_frames {
                self.emit_one_frame(cfg);
            }
        }
    }
}

// =============================================================================
// Free helpers
// =============================================================================

fn pick_ext(available: bool, ext: f32, port: f32) -> f32 {
    if available {
        ext
    } else {
        port
    }
}

fn pick_ext_nonneg(available: bool, ext: f32, port: f32) -> f32 {
    if available && ext >= 0.0 {
        ext
    } else {
        port
    }
}

#[inline]
fn apply_gain(spectrum: &mut [(f32, f32); NUM_FREQ_BINS], gain: f32) {
    for bin in spectrum.iter_mut() {
        bin.0 *= gain;
        bin.1 *= gain;
    }
}

/// Linear-blend `original` and `enhanced` by `strength` (0 = original,
/// 1 = enhanced).
fn blend_strength(
    original: &[(f32, f32); NUM_FREQ_BINS],
    enhanced: &[(f32, f32); NUM_FREQ_BINS],
    strength: f32,
) -> [(f32, f32); NUM_FREQ_BINS] {
    if strength <= 0.0 {
        return *original;
    }
    if strength >= 1.0 {
        return *enhanced;
    }
    let mut blended = *original;
    let inv = 1.0 - strength;
    for i in 0..NUM_FREQ_BINS {
        blended[i].0 = original[i].0 * inv + enhanced[i].0 * strength;
        blended[i].1 = original[i].1 * inv + enhanced[i].1 * strength;
    }
    blended
}

// =============================================================================
// LADSPA Plugin trait impl
// =============================================================================

impl Plugin for GtcrnPlugin {
    fn activate(&mut self) {}

    fn deactivate(&mut self) {}

    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input = ports[PORT_INPUT].unwrap_audio();
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut();

        let controls = self.resolve_controls(ports);

        // Bypass: enable off OR strength at zero.
        if controls.enable < 0.5 || controls.strength <= 0.0 {
            output[..sample_count].copy_from_slice(&input[..sample_count]);
            return;
        }

        let requested = ModelType::from_control(controls.model);
        self.update_primary_model(requested);
        self.update_secondary_model(controls.model_blend);

        let strength = controls.strength.clamp(0.0, 1.0);
        let speech_strength = controls.speech_strength.clamp(0.0, 1.0);
        let new_la = self.ms_to_frames(controls.lookahead_ms.clamp(0.0, 200.0));
        if new_la != self.lookahead_frames {
            self.lookahead_frames = new_la;
        }

        self.feed_input(input, sample_count);

        let cfg = ProcessConfig {
            noise_strength: strength,
            speech_strength,
            model_blend: controls.model_blend,
            voice_recovery: controls.voice_recovery.clamp(0.0, 1.0),
        };
        self.process_frames(&cfg);
        self.emit_output(&mut output, sample_count);

        apply_gate_pass(
            &mut self.gate,
            self.sample_rate,
            self.vad_gate,
            input,
            &mut output,
            sample_count,
            ports,
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ms_to_frames_calc(hop_size: usize, sample_rate: f32, ms: f32) -> usize {
        if ms <= 0.0 {
            return 0;
        }
        let frame_ms = (hop_size as f32) / sample_rate * 1000.0;
        (ms / frame_ms).round() as usize
    }

    #[test]
    fn ms_to_frames_48khz() {
        // At 48 kHz, hop=768 → frame_ms ≈ 16 ms.
        assert_eq!(ms_to_frames_calc(768, 48000.0, 0.0), 0);
        assert_eq!(ms_to_frames_calc(768, 48000.0, 16.0), 1);
        assert_eq!(ms_to_frames_calc(768, 48000.0, 50.0), 3);
        assert_eq!(ms_to_frames_calc(768, 48000.0, 200.0), 13);
    }

    #[test]
    fn ms_to_frames_44100() {
        assert_eq!(ms_to_frames_calc(706, 44100.0, 0.0), 0);
        let frames_50 = ms_to_frames_calc(706, 44100.0, 50.0);
        assert!(
            frames_50 == 3 || frames_50 == 4,
            "expected 3 or 4, got {frames_50}"
        );
    }

    #[test]
    fn ms_to_frames_negative() {
        assert_eq!(ms_to_frames_calc(768, 48000.0, -10.0), 0);
    }

    #[test]
    fn ms_to_frames_96khz() {
        assert_eq!(ms_to_frames_calc(1536, 96000.0, 50.0), 3);
    }

    #[test]
    fn pick_ext_uses_external_when_available() {
        assert!((pick_ext(true, 0.7, 0.0) - 0.7).abs() < 1e-6);
        assert!((pick_ext(false, 0.7, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn pick_ext_nonneg_falls_back_for_sentinel() {
        assert!((pick_ext_nonneg(true, -1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((pick_ext_nonneg(true, 0.3, 0.5) - 0.3).abs() < 1e-6);
        assert!((pick_ext_nonneg(false, 0.3, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blend_strength_endpoints_no_alloc() {
        let orig = [(1.0, 2.0); NUM_FREQ_BINS];
        let enh = [(3.0, 4.0); NUM_FREQ_BINS];
        let zero = blend_strength(&orig, &enh, 0.0);
        let one = blend_strength(&orig, &enh, 1.0);
        assert!((zero[0].0 - 1.0).abs() < 1e-6);
        assert!((one[0].0 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn blend_strength_midpoint() {
        let orig = [(0.0, 0.0); NUM_FREQ_BINS];
        let enh = [(2.0, 4.0); NUM_FREQ_BINS];
        let mid = blend_strength(&orig, &enh, 0.5);
        assert!((mid[0].0 - 1.0).abs() < 1e-6);
        assert!((mid[0].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn apply_gain_scales_in_place() {
        let mut s = [(1.0, 2.0); NUM_FREQ_BINS];
        apply_gain(&mut s, 2.0);
        assert!((s[0].0 - 2.0).abs() < 1e-6);
        assert!((s[0].1 - 4.0).abs() < 1e-6);
    }

    #[test]
    fn buffers_pre_sized_for_max_block() {
        // The largest practical host block we expect; constructor must
        // size the input/output accumulators above this so a typical
        // run() never reallocates.
        const { assert!(MAX_HOST_BLOCK >= 16 * 1024) };
    }

    #[test]
    fn spectrum_pool_capacity_is_generous() {
        // Lookahead tops out near 200 ms / 16 ms ≈ 13 frames; the pool
        // must hold significantly more than that to absorb bursts.
        const { assert!(SPECTRUM_POOL_CAPACITY >= 16) };
    }
}
