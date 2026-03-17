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

use std::collections::VecDeque;

use realfft::num_complex::Complex;

use crate::biquad::Biquad;
use crate::model::{GtcrnModel, ModelType, NUM_FREQ_BINS};
use crate::stft::StftProcessor;
use crate::{
    PORT_ENABLE, PORT_INPUT, PORT_LOOKAHEAD_MS, PORT_MODEL, PORT_MODEL_BLEND,
    PORT_OUTPUT, PORT_SPEECH_STRENGTH, PORT_STRENGTH, PORT_VOICE_RECOVERY,
};
use ladspa::{Plugin, PluginDescriptor, PortConnection};

// =============================================================================
// Process Configuration
// =============================================================================

/// All per-frame processing parameters collected into a single struct.
struct ProcessConfig {
    noise_strength: f32,
    speech_strength: f32,
    model_blend: bool,
    /// Voice recovery level (0.0 = cut all >8kHz, 1.0 = full original).
    voice_recovery: f32,
}
// =============================================================================
// Constants
// =============================================================================

/// Highpass pre-filter cutoff frequency in Hz.
/// Removes sub-audible content that confuses the GTCRN model.
const HP_CUTOFF_HZ: f32 = 80.0;

/// VAD decay smoothing coefficient (EMA).
/// Higher = slower decay, more stable gate. `0.92` ≈ 190 ms time constant at 62 fps.
const VAD_DECAY_COEFF: f32 = 0.92;

/// Number of `run()` calls with model_blend=0 before unloading the secondary model.
/// At ~62 frames/sec (48 kHz, hop=768) this is ≈1.6 seconds.
const DUAL_MODEL_UNLOAD_FRAMES: usize = 100;

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
    speech_strength: f32,
    lookahead_ms: f32,
    model_blend: f32,
    voice_recovery: f32,
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
            speech_strength: -1.0,
            lookahead_ms: -1.0,
            model_blend: 1.0,
            voice_recovery: -1.0,
            available: false,
            counter: EXTERNAL_POLL_INTERVAL, // trigger immediate read
        };
        ctrl.poll();
        ctrl
    }

    /// Test-only constructor with a custom file path.
    #[cfg(test)]
    fn with_path(path: std::path::PathBuf) -> Self {
        Self {
            path,
            strength: -1.0,
            model_type: -1.0,
            speech_strength: -1.0,
            lookahead_ms: -1.0,
            model_blend: 1.0,
            voice_recovery: -1.0,
            available: false,
            counter: EXTERNAL_POLL_INTERVAL,
        }
    }

    /// Re-read the control file if enough calls have elapsed.
    fn poll(&mut self) {
        self.counter += 1;
        if self.counter < EXTERNAL_POLL_INTERVAL {
            return;
        }
        self.counter = 0;
        if let Ok(data) = std::fs::read(&self.path) {
            // Format v7: 24 bytes (6 floats)
            // [strength, model, speech_strength, lookahead_ms,
            //  model_blend, voice_recovery]
            if data.len() >= 8 {
                self.strength = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                self.model_type = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                self.available = true;
            }
            if data.len() >= 16 {
                self.speech_strength = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                self.lookahead_ms = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
            }
            if data.len() >= 24 {
                self.model_blend = f32::from_le_bytes([data[16], data[17], data[18], data[19]]);
                self.voice_recovery = f32::from_le_bytes([data[20], data[21], data[22], data[23]]);
            }
        }
    }
}

// =============================================================================
// Buffered Frame (for lookahead)
// =============================================================================

/// Stores a processed STFT frame along with its VAD decision,
/// enabling look-ahead before choosing the blend strength.
struct BufferedFrame {
    original: [(f32, f32); NUM_FREQ_BINS],
    enhanced: [(f32, f32); NUM_FREQ_BINS],
    /// Secondary model output for dual-model blending (VCTK during speech)
    enhanced_alt: [(f32, f32); NUM_FREQ_BINS],
    /// Whether dual-model blending is active for this frame
    dual_model: bool,
    is_speech: bool,
    /// Original full spectrum (all n_bins) for HF reconstruction.
    original_full_spectrum: Vec<Complex<f32>>,
}

// =============================================================================
// Plugin
// =============================================================================

/// GTCRN LADSPA plugin instance with VAD-driven dual-strength and lookahead.
///
/// All processing happens synchronously within `run()`. STFT size adapts
/// to the host sample rate so no sample-rate conversion is needed.
///
/// The ONNX model is lazily initialized on first `run()` call.
pub struct GtcrnPlugin {
    model: Option<GtcrnModel>,
    second_model: Option<GtcrnModel>,
    stft: StftProcessor,

    /// Host sample rate in Hz
    sample_rate: f32,

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

    // -------------------------------------------------------------------------
    // VAD + Dual-Strength State
    // -------------------------------------------------------------------------
    vad_gate: f32,
    vad_energy_smooth: f32,
    lookahead_buf: VecDeque<BufferedFrame>,
    lookahead_frames: usize,

    // -------------------------------------------------------------------------
    // Feature state
    // -------------------------------------------------------------------------
    /// Model blending — counter for lazy unload
    model_blend_zero_count: usize,

    /// Input normalization: running RMS for boosting quiet audio before model
    input_norm_rms: f32,

    /// Input-based onset detector: noise floor EMA (energy of input during silence)
    input_noise_floor: f32,
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
            second_model: None,
            stft,
            sample_rate: sample_rate as f32,
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
            vad_gate: 0.0,
            vad_energy_smooth: 0.0,
            lookahead_buf: VecDeque::with_capacity(8),
            lookahead_frames: 0,
            model_blend_zero_count: 0,
            input_norm_rms: 0.0,
            input_noise_floor: 0.0,
        })
    }

    /// Convert milliseconds to STFT frames.
    fn ms_to_frames(&self, ms: f32) -> usize {
        if ms <= 0.0 {
            return 0;
        }
        let frame_ms = (self.hop_size as f32) / self.sample_rate * 1000.0;
        (ms / frame_ms).round() as usize
    }

    /// Process complete STFT frames with all NOISE2 feature processing.
    fn process_frames(&mut self, cfg: &ProcessConfig) {
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

            // Copy spectrum for model input (needed to avoid borrow conflicts)
            let mut model_input = self.spectrum_buffer;

            // ── Input normalization: boost quiet audio before model ─────
            // The model was trained on typical speech levels. Quiet audio
            // has magnitudes too small for the model to distinguish from
            // noise, causing over-suppression. We normalize to a reference
            // level before inference and undo after.
            const NORM_TARGET_RMS: f32 = 0.5;
            const NORM_MIN_RMS: f32 = 0.02;
            const NORM_MAX_GAIN: f32 = 3.0;
            const NORM_EMA: f32 = 0.15;

            let frame_rms = {
                let mag_sq_sum: f32 = self
                    .spectrum_buffer
                    .iter()
                    .map(|&(re, im)| re * re + im * im)
                    .sum();
                (mag_sq_sum / NUM_FREQ_BINS as f32).sqrt()
            };
            self.input_norm_rms = if self.input_norm_rms < 1e-6 {
                frame_rms
            } else {
                self.input_norm_rms * (1.0 - NORM_EMA) + frame_rms * NORM_EMA
            };
            let norm_gain =
                if self.input_norm_rms > NORM_MIN_RMS && self.input_norm_rms < NORM_TARGET_RMS {
                    (NORM_TARGET_RMS / self.input_norm_rms).min(NORM_MAX_GAIN)
                } else {
                    1.0
                };
            if norm_gain > 1.01 {
                for bin in &mut model_input {
                    bin.0 *= norm_gain;
                    bin.1 *= norm_gain;
                }
            }

            // Model inference (always at full power — blend ratio varies)
            let mut enhanced = match self.model.as_mut() {
                Some(m) => m.process_frame(&model_input).unwrap_or_else(|e| {
                    eprintln!("GTCRN model error: {e}");
                    self.spectrum_buffer
                }),
                None => self.spectrum_buffer,
            };

            // Undo input normalization on model output
            if norm_gain > 1.01 {
                let inv_gain = 1.0 / norm_gain;
                for bin in &mut enhanced {
                    bin.0 *= inv_gain;
                    bin.1 *= inv_gain;
                }
            }

            // --- Feature 10: Dual-Model Blending ---
            // When enabled, both DNS3 (restrictive) and VCTK (gentle) run.
            // DNS3 output is used during silence (better noise gate).
            // VCTK output is used during speech (preserves voice quality).
            let mut enhanced_alt = [(0.0f32, 0.0f32); NUM_FREQ_BINS];
            let dual_model = cfg.model_blend;
            if dual_model {
                if let Some(ref mut second) = self.second_model {
                    enhanced_alt = second
                        .process_frame(&model_input)
                        .unwrap_or(self.spectrum_buffer);
                    // Undo input normalization on second model output
                    if norm_gain > 1.01 {
                        let inv_gain = 1.0 / norm_gain;
                        for bin in &mut enhanced_alt {
                            bin.0 *= inv_gain;
                            bin.1 *= inv_gain;
                        }
                    }
                    // Ensure DNS3 is in `enhanced` and VCTK in `enhanced_alt`
                    let primary_is_dns3 = self
                        .model
                        .as_ref()
                        .map(|m| m.model_type() == ModelType::Dns3)
                        .unwrap_or(true);
                    if !primary_is_dns3 {
                        // Primary is VCTK, secondary is DNS3 — swap so enhanced=DNS3
                        std::mem::swap(&mut enhanced, &mut enhanced_alt);
                    }
                }
            }

            // Compute VAD from spectral energy ratio with EMA smoothing
            // Always use DNS3 output (enhanced) for VAD — stronger model = cleaner signal
            let input_energy: f32 = self
                .spectrum_buffer
                .iter()
                .map(|&(re, im)| re * re + im * im)
                .sum();
            let output_energy: f32 = enhanced.iter().map(|&(re, im)| re * re + im * im).sum();
            let energy_ratio = if input_energy > 1e-10 {
                output_energy / input_energy
            } else {
                0.0
            };

            const VAD_EMA_ATTACK: f32 = 0.15;
            self.vad_energy_smooth =
                self.vad_energy_smooth * (1.0 - VAD_EMA_ATTACK) + energy_ratio * VAD_EMA_ATTACK;
            const VAD_THRESHOLD: f32 = 0.05;
            let is_speech = self.vad_energy_smooth > VAD_THRESHOLD;

            // Save original full spectrum for this frame (before synthesize overwrites it)
            let orig_spectrum = self.stft.original_spectrum().to_vec();

            // Push to lookahead buffer
            self.lookahead_buf.push_back(BufferedFrame {
                original: self.spectrum_buffer,
                enhanced,
                enhanced_alt,
                dual_model,
                is_speech,
                original_full_spectrum: orig_spectrum,
            });

            // Emit frames when buffer exceeds lookahead depth
            while self.lookahead_buf.len() > self.lookahead_frames {
                let frame = self.lookahead_buf.pop_front().unwrap();
                let any_future_speech = self.lookahead_buf.iter().any(|f| f.is_speech);

                // ── Input-energy onset detector ────────────────────────
                // Compute input energy directly from the original spectrum
                // (independent of model output). This breaks the feedback
                // loop where model attenuation → low VAD → more suppression.
                let input_energy: f32 = frame
                    .original
                    .iter()
                    .map(|&(re, im)| re * re + im * im)
                    .sum();

                // Update noise floor during confirmed silence
                if self.vad_gate < 0.1 && !frame.is_speech {
                    const FLOOR_ALPHA: f32 = 0.05;
                    if self.input_noise_floor < 1e-10 {
                        self.input_noise_floor = input_energy;
                    } else {
                        self.input_noise_floor = self.input_noise_floor * (1.0 - FLOOR_ALPHA)
                            + input_energy * FLOOR_ALPHA;
                    }
                }

                // If input energy is significantly above noise floor,
                // this is speech onset — force vad_gate high immediately
                let onset_detected =
                    self.input_noise_floor > 1e-10 && input_energy > self.input_noise_floor * 4.0;

                // Update VAD gate
                if frame.is_speech || any_future_speech || onset_detected {
                    self.vad_gate = 1.0;
                } else {
                    self.vad_gate *= VAD_DECAY_COEFF;
                    if self.vad_gate < 0.01 {
                        self.vad_gate = 0.0;
                    }
                }

                // Effective strength: interpolate between noise and speech strengths
                let effective_strength = if self.vad_gate > 0.0 {
                    cfg.noise_strength * (1.0 - self.vad_gate) + cfg.speech_strength * self.vad_gate
                } else {
                    cfg.noise_strength
                };

                // Choose enhanced output: dual-model crossfades DNS3↔VCTK by VAD
                let chosen_enhanced = if frame.dual_model {
                    // Crossfade: silence→DNS3 (enhanced), speech→VCTK (enhanced_alt)
                    let speech_mix = ((self.vad_gate - 0.2) / 0.5).clamp(0.0, 1.0);
                    if speech_mix <= 0.0 {
                        frame.enhanced // Pure DNS3 during silence
                    } else if speech_mix >= 1.0 {
                        frame.enhanced_alt // Pure VCTK during speech
                    } else {
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
                } else {
                    frame.enhanced
                };

                // Blend original and chosen enhanced spectra
                let blend_strength = effective_strength;
                let final_spectrum = if blend_strength <= 0.0 {
                    frame.original
                } else if blend_strength >= 1.0 {
                    chosen_enhanced
                } else {
                    let mut blended = frame.original;
                    let inv = 1.0 - blend_strength;
                    for i in 0..NUM_FREQ_BINS {
                        blended[i].0 =
                            frame.original[i].0 * inv + chosen_enhanced[i].0 * blend_strength;
                        blended[i].1 =
                            frame.original[i].1 * inv + chosen_enhanced[i].1 * blend_strength;
                    }
                    blended
                };

                // iSTFT synthesis — HighBandProcessor handles HF internally
                // voice_recovery controls HF reconstruction: 0.0 = cut all HF,
                // 1.0 = full reconstruction.
                let synth = self.stft.synthesize(
                    &final_spectrum,
                    &frame.original_full_spectrum,
                    self.vad_gate,
                    cfg.voice_recovery,
                );
                let proc_len = synth.len();

                // Copy to output accumulator
                let processed_start = self.output_accum_len;
                self.output_accum[processed_start..processed_start + proc_len]
                    .copy_from_slice(synth);

                self.output_accum_len += proc_len;
            }
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
        let speech_strength_control = *ports[PORT_SPEECH_STRENGTH].unwrap_control();
        let lookahead_ms_control = *ports[PORT_LOOKAHEAD_MS].unwrap_control();
        let model_blend_control = *ports[PORT_MODEL_BLEND].unwrap_control();
        let voice_recovery_control = *ports[PORT_VOICE_RECOVERY].unwrap_control();

        // Poll external control file for live parameter updates
        self.ext_controls.poll();

        // External file overrides LADSPA port values when available
        let (strength_val, model_val) = if self.ext_controls.available {
            (self.ext_controls.strength, self.ext_controls.model_type)
        } else {
            (strength_control, model_control)
        };

        let speech_strength_val =
            if self.ext_controls.available && self.ext_controls.speech_strength >= 0.0 {
                self.ext_controls.speech_strength
            } else {
                speech_strength_control
            };
        let lookahead_ms_val =
            if self.ext_controls.available && self.ext_controls.lookahead_ms >= 0.0 {
                self.ext_controls.lookahead_ms
            } else {
                lookahead_ms_control
            };
        let model_blend_val = if self.ext_controls.available {
            self.ext_controls.model_blend
        } else {
            model_blend_control
        };
        let voice_recovery_val =
            if self.ext_controls.available && self.ext_controls.voice_recovery >= 0.0 {
                self.ext_controls.voice_recovery
            } else {
                voice_recovery_control
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
                if let Some(m) = self.model.as_mut() {
                    m.set_model_type(requested);
                }
            }
            _ => {}
        }

        // Dual-Model Blending — manage secondary model lifecycle
        let model_blend = model_blend_val >= 0.5;
        if model_blend {
            self.model_blend_zero_count = 0;
            let primary_type = self
                .model
                .as_ref()
                .map(|m| m.model_type())
                .unwrap_or(ModelType::Dns3);
            let secondary_type = match primary_type {
                ModelType::Dns3 => ModelType::Vctk,
                ModelType::Vctk => ModelType::Dns3,
            };
            if self.second_model.is_none() {
                self.second_model = Some(GtcrnModel::new(secondary_type));
            }
        } else {
            self.model_blend_zero_count += 1;
            if self.model_blend_zero_count > DUAL_MODEL_UNLOAD_FRAMES && self.second_model.is_some()
            {
                self.second_model = None;
            }
        }

        let strength = strength_val.clamp(0.0, 1.0);
        let speech_strength = speech_strength_val.clamp(0.0, 1.0);

        // Update lookahead if changed
        let new_la_frames = self.ms_to_frames(lookahead_ms_val.clamp(0.0, 200.0));
        if new_la_frames != self.lookahead_frames {
            self.lookahead_frames = new_la_frames;
        }

        // Append new input to accumulator (after HP pre-filter)
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

        // Build process config
        let cfg = ProcessConfig {
            noise_strength: strength,
            speech_strength,
            model_blend,
            voice_recovery: voice_recovery_val.clamp(0.0, 1.0),
        };

        // Process pipeline
        self.process_frames(&cfg);

        // Copy available output to host buffer
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // =========================================================================
    // ms_to_frames
    // =========================================================================

    /// Helper: compute ms_to_frames with given hop_size and sample_rate
    fn ms_to_frames(hop_size: usize, sample_rate: f32, ms: f32) -> usize {
        if ms <= 0.0 {
            return 0;
        }
        let frame_ms = (hop_size as f32) / sample_rate * 1000.0;
        (ms / frame_ms).round() as usize
    }

    #[test]
    fn ms_to_frames_48khz() {
        // At 48 kHz, hop=768 → frame_ms ≈ 16 ms
        assert_eq!(ms_to_frames(768, 48000.0, 0.0), 0);
        assert_eq!(ms_to_frames(768, 48000.0, 16.0), 1);
        assert_eq!(ms_to_frames(768, 48000.0, 50.0), 3);
        assert_eq!(ms_to_frames(768, 48000.0, 200.0), 13); // 200 / 16 = 12.5 → 13
    }

    #[test]
    fn ms_to_frames_44100() {
        // At 44.1 kHz, nfft=1412, hop=706 → frame_ms ≈ 16.01 ms
        assert_eq!(ms_to_frames(706, 44100.0, 0.0), 0);
        let frames_50 = ms_to_frames(706, 44100.0, 50.0);
        assert!(
            frames_50 == 3 || frames_50 == 4,
            "expected 3 or 4, got {frames_50}"
        );
    }

    #[test]
    fn ms_to_frames_negative() {
        assert_eq!(ms_to_frames(768, 48000.0, -10.0), 0);
    }

    #[test]
    fn ms_to_frames_96khz() {
        // At 96 kHz, hop=1536 → frame_ms = 16 ms (same as 48k due to proportional scaling)
        assert_eq!(ms_to_frames(1536, 96000.0, 50.0), 3);
    }

    // =========================================================================
    // ExternalControls::poll — byte format parsing
    // =========================================================================

    fn write_control_file(path: &std::path::Path, data: &[u8]) {
        let mut file = std::fs::File::create(path).expect("create control file");
        file.write_all(data).expect("write control file");
    }

    #[test]
    fn poll_v2_format_8_bytes() {
        let dir = tempdir();
        let path = dir.join("controls");
        let strength: f32 = 0.75;
        let model: f32 = 1.0;
        let mut data = Vec::new();
        data.extend_from_slice(&strength.to_le_bytes());
        data.extend_from_slice(&model.to_le_bytes());
        write_control_file(&path, &data);

        let mut ctrl = ExternalControls::with_path(path);
        ctrl.poll();
        assert!(ctrl.available);
        assert!((ctrl.strength - 0.75).abs() < 1e-6);
        assert!((ctrl.model_type - 1.0).abs() < 1e-6);
        // Extended fields remain at defaults
        assert_eq!(ctrl.speech_strength, -1.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poll_v4_format_16_bytes() {
        let dir = tempdir();
        let path = dir.join("controls");
        let vals: [f32; 4] = [0.5, 0.0, 0.8, 100.0];
        let mut data = Vec::new();
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        write_control_file(&path, &data);

        let mut ctrl = ExternalControls::with_path(path);
        ctrl.poll();
        assert!(ctrl.available);
        assert!((ctrl.strength - 0.5).abs() < 1e-6);
        assert!((ctrl.model_type - 0.0).abs() < 1e-6);
        assert!((ctrl.speech_strength - 0.8).abs() < 1e-6);
        assert!((ctrl.lookahead_ms - 100.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poll_v7_format_24_bytes() {
        let dir = tempdir();
        let path = dir.join("controls");
        let vals: [f32; 6] = [0.9, 0.0, 1.0, 50.0, 0.5, 0.0];
        let mut data = Vec::new();
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        write_control_file(&path, &data);

        let mut ctrl = ExternalControls::with_path(path);
        ctrl.poll();
        assert!(ctrl.available);
        assert!((ctrl.strength - 0.9).abs() < 1e-6);
        assert!((ctrl.model_type - 0.0).abs() < 1e-6);
        assert!((ctrl.speech_strength - 1.0).abs() < 1e-6);
        assert!((ctrl.lookahead_ms - 50.0).abs() < 1e-6);
        assert!((ctrl.voice_enhance - 0.5).abs() < 1e-6);
        assert!((ctrl.model_blend - 0.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poll_missing_file() {
        let path = std::path::PathBuf::from("/tmp/gtcrn-test-nonexistent-12345");
        let mut ctrl = ExternalControls::with_path(path);
        ctrl.poll();
        assert!(!ctrl.available);
    }

    #[test]
    fn poll_short_file_ignored() {
        let dir = tempdir();
        let path = dir.join("controls");
        write_control_file(&path, &[0u8; 4]); // Only 4 bytes — not enough

        let mut ctrl = ExternalControls::with_path(path);
        ctrl.poll();
        assert!(!ctrl.available); // 4 bytes < 8, so not parsed
        std::fs::remove_dir_all(&dir).ok();
    }

    // =========================================================================
    // Constants sanity checks
    // =========================================================================

    #[test]
    fn constants_sane() {
        const {
            assert!(HP_CUTOFF_HZ > 0.0);
            assert!(VAD_DECAY_COEFF > 0.0 && VAD_DECAY_COEFF < 1.0);
            assert!(MAX_GAIN_COMP_RATIO > 1.0);
            assert!(MAX_DEESS_REDUCTION > 0.0 && MAX_DEESS_REDUCTION < 1.0);
            assert!(DEESS_CENTER_BIN > 0.0);
            assert!(DEESS_WIDTH_BINS > 0.0);
            assert!(SMOOTHING_FLOOR_FACTOR > 0.0 && SMOOTHING_FLOOR_FACTOR < 1.0);
            assert!(DUAL_MODEL_UNLOAD_FRAMES > 0);
        }
    }

    /// Create a temporary directory for test files.
    fn tempdir() -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gtcrn-test-{id}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
