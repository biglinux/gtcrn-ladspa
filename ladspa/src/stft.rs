//! Real-time STFT/iSTFT processing with overlap-add synthesis.
//!
//! Adapts the FFT size to the host sample rate so that the first 257 bins
//! always cover 0–8 kHz (matching the 16 kHz model's frequency range).
//! The `sqrt(hann)` window ensures perfect reconstruction with 50% overlap.
//!
//! ## Split-Band Architecture
//!
//! The neural network only processes 0–8 kHz (257 bins). Frequencies above
//! 8 kHz are handled by [`HighBandProcessor`] using a transient-aware
//! spectral gate (when HF SNR is adequate) or an "air exciter" that
//! synthesizes harmonics from the clean low band (when HF is too noisy).
//! A 4-bin raised-cosine crossfade at the boundary ensures phase continuity.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::model::NUM_FREQ_BINS;

/// FFT size the GTCRN model was trained with (at 16 kHz).
const MODEL_NFFT: f64 = 512.0;

/// Sample rate the model was trained at.
const MODEL_SR: f64 = 16_000.0;

/// Number of crossfade bins at the low/high band boundary (centered on bin 257).
const CROSSFADE_BINS: usize = 4;

// =============================================================================
// High-Frequency Band Processor
// =============================================================================

/// Processes frequencies above 8 kHz that the neural network cannot handle.
///
/// Uses a **Transient-Aware Spectral Gate** that passes through original HF
/// content during speech while gating it during silence. Spectral flux
/// detects sharp transients ("T", "S", "K") and bypasses the gate for ~20 ms
/// to preserve consonant crispness.
///
/// When the original HF has poor SNR (too noisy to pass through), the
/// processor falls back to an **Air Exciter** that synthesizes clean
/// harmonics from the enhanced low-band spectrum.
pub struct HighBandProcessor {
    /// Previous frame HF magnitudes for spectral flux (transient detection).
    prev_hf_mag: Vec<f32>,
    /// Whether `prev_hf_mag` contains valid data from a prior frame.
    has_prev: bool,
    /// Remaining frames to hold gate bypass after a detected transient.
    transient_hold: usize,
    /// Duration of transient bypass in STFT frames (~20 ms).
    transient_hold_frames: usize,
    /// Adaptive noise floor estimate for each HF bin.
    hf_noise_floor: Vec<f32>,
    /// Whether the noise floor has been initialized during a silence period.
    hf_noise_initialized: bool,
}

impl HighBandProcessor {
    /// Creates a processor for the given number of HF bins.
    fn new(hf_bin_count: usize, hop_size: usize, sample_rate: u32) -> Self {
        let frame_dur_s = hop_size as f64 / sample_rate as f64;
        let hold_frames = (0.020 / frame_dur_s).ceil() as usize;

        Self {
            prev_hf_mag: vec![0.0; hf_bin_count],
            has_prev: false,
            transient_hold: 0,
            transient_hold_frames: hold_frames.max(1),
            hf_noise_floor: vec![1e-6; hf_bin_count],
            hf_noise_initialized: false,
        }
    }

    /// Main entry point: processes HF bins and writes result to `output`.
    ///
    /// Decides between the transient-aware spectral gate (adequate HF SNR)
    /// and the air exciter synthesis (noisy HF) based on estimated SNR.
    ///
    /// * `hf_original` – original bins 257..n_bins from the analysis frame.
    /// * `enhanced_low` – clean 257-bin spectrum from the neural network.
    /// * `vad_probability` – voice activity gate (0.0 = silence, 1.0 = speech).
    /// * `external_transient` – forces gate bypass regardless of flux detection.
    /// * `output` – destination slice (same length as `hf_original`).
    pub fn process_high_band(
        &mut self,
        hf_original: &[Complex<f32>],
        enhanced_low: &[(f32, f32); NUM_FREQ_BINS],
        vad_probability: f32,
        external_transient: bool,
        output: &mut [Complex<f32>],
    ) {
        self.update_noise_floor(hf_original, vad_probability);

        let flux_transient = self.detect_transient(hf_original);
        if flux_transient || external_transient {
            self.transient_hold = self.transient_hold_frames;
        } else if self.transient_hold > 0 {
            self.transient_hold -= 1;
        }

        let hf_snr = self.estimate_hf_snr(hf_original);

        if hf_snr < 2.0 && self.hf_noise_initialized {
            self.synthesize_air(enhanced_low, output);
        } else {
            self.spectral_gate(hf_original, vad_probability, output);
        }
    }

    /// Detects transient onsets via half-wave-rectified spectral flux.
    ///
    /// A transient ("T", "S", "K") produces a sharp positive flux spike
    /// across many HF bins simultaneously.
    fn detect_transient(&mut self, hf: &[Complex<f32>]) -> bool {
        let count = hf.len().min(self.prev_hf_mag.len());
        if count == 0 {
            return false;
        }

        let mut flux = 0.0f32;
        let mut avg_mag = 0.0f32;
        for (h, prev) in hf[..count].iter().zip(self.prev_hf_mag[..count].iter_mut()) {
            let mag = h.norm();
            avg_mag += mag;
            let diff = mag - *prev;
            if diff > 0.0 {
                flux += diff;
            }
            *prev = mag;
        }
        avg_mag /= count as f32;

        let was_valid = self.has_prev;
        self.has_prev = true;

        if !was_valid || avg_mag < 1e-10 {
            return false;
        }

        // Normalised flux: fraction of total energy that is "new"
        flux / (avg_mag * count as f32) > 0.35
    }

    /// Transient-aware spectral gate driven by VAD probability.
    ///
    /// During speech: HF passes with per-bin SNR gating.
    /// During silence: HF fully attenuated.
    /// During transient hold (~20 ms): gate bypassed for consonant crispness.
    fn spectral_gate(
        &self,
        hf: &[Complex<f32>],
        vad_probability: f32,
        output: &mut [Complex<f32>],
    ) {
        let gate_bypassed = self.transient_hold > 0;
        let envelope = if gate_bypassed || vad_probability > 0.7 {
            1.0
        } else if vad_probability > 0.1 {
            (vad_probability - 0.1) / 0.6
        } else {
            0.0
        };

        let count = hf.len().min(output.len());

        if self.hf_noise_initialized {
            for i in 0..count {
                let snr = hf[i].norm() / (self.hf_noise_floor[i] + 1e-10);
                let snr_gain = ((snr - 1.5) / 3.5).clamp(0.0, 1.0);
                let gain = if gate_bypassed {
                    snr_gain.max(0.5)
                } else {
                    envelope * snr_gain
                };
                output[i] = Complex::new(hf[i].re * gain, hf[i].im * gain);
            }
        } else {
            let atten = envelope * 0.5;
            for i in 0..count {
                output[i] = Complex::new(hf[i].re * atten, hf[i].im * atten);
            }
        }

        for o in output.iter_mut().skip(count) {
            *o = Complex::new(0.0, 0.0);
        }
    }

    /// Synthesises clean HF from the enhanced low band (air exciter).
    ///
    /// Mirrors 4–8 kHz content (bins 128–256) into 8–16 kHz with 2nd-harmonic
    /// generation and a ~6 dB/octave tilt EQ for natural rolloff.
    fn synthesize_air(
        &self,
        enhanced_low: &[(f32, f32); NUM_FREQ_BINS],
        output: &mut [Complex<f32>],
    ) {
        let hf_count = output.len();
        output.fill(Complex::new(0.0, 0.0));

        let src_start = 128;
        let src_len = NUM_FREQ_BINS - src_start; // 129 bins (4–8 kHz)

        // Reference: average magnitude at 6–8 kHz
        let ref_energy: f32 = enhanced_low[192..NUM_FREQ_BINS]
            .iter()
            .map(|&(re, im)| (re * re + im * im).sqrt())
            .sum::<f32>()
            / (NUM_FREQ_BINS - 192) as f32;

        if ref_energy < 1e-10 {
            return;
        }

        // Synthesise up to ~16 kHz (256 HF bins ≈ 8 kHz span at ~31 Hz/bin)
        let target_bins = 256.min(hf_count);

        for (i, out) in output.iter_mut().enumerate().take(target_bins) {
            let src_idx = src_start + (i % src_len);
            let (re, im) = enhanced_low[src_idx];
            let mag = (re * re + im * im).sqrt();

            // 2nd harmonic: square the magnitude
            let harmonic_mag = mag * mag;

            // Tilt EQ: −6 dB/octave above 8 kHz
            let freq_ratio = 1.0 + i as f32 * 31.25 / 8000.0;
            let tilt = 1.0 / freq_ratio;

            let phase = im.atan2(re);
            let final_mag = harmonic_mag * tilt;

            *out = Complex::new(final_mag * phase.cos(), final_mag * phase.sin());
        }

        // Normalise against clean 6–8 kHz reference
        let synth_energy: f32 = output
            .iter()
            .take(target_bins)
            .map(|c| c.norm())
            .sum::<f32>()
            / target_bins.max(1) as f32;

        if synth_energy > 1e-10 {
            let ratio = (ref_energy * 0.4 / synth_energy).min(3.0);
            for c in output.iter_mut().take(target_bins) {
                c.re *= ratio;
                c.im *= ratio;
            }
        }
    }

    /// Updates adaptive HF noise floor during silence periods.
    fn update_noise_floor(&mut self, hf: &[Complex<f32>], vad: f32) {
        if vad >= 0.1 {
            return;
        }
        let count = hf.len().min(self.hf_noise_floor.len());
        for (h, floor) in hf[..count]
            .iter()
            .zip(self.hf_noise_floor[..count].iter_mut())
        {
            let mag = h.norm();
            if self.hf_noise_initialized {
                *floor = 0.99 * *floor + 0.01 * mag;
            } else {
                *floor = mag;
            }
        }
        self.hf_noise_initialized = true;
    }

    /// Estimates average HF signal-to-noise ratio.
    fn estimate_hf_snr(&self, hf: &[Complex<f32>]) -> f32 {
        if !self.hf_noise_initialized {
            return 10.0; // Assume good SNR until calibrated
        }
        let count = hf.len().min(self.hf_noise_floor.len());
        if count == 0 {
            return 0.0;
        }
        let total: f32 = hf
            .iter()
            .enumerate()
            .take(count)
            .map(|(i, c)| c.norm() / (self.hf_noise_floor[i] + 1e-10))
            .sum();
        total / count as f32
    }
}

// =============================================================================
// STFT Processor
// =============================================================================

/// STFT processor with parameters derived from the host sample rate.
///
/// Computes `NFFT = round(512 × sr / 16000)` (even) so that the first 257
/// bins always cover 0–8 kHz. `HOP = NFFT / 2` (50% overlap).
/// Spectrum is scaled by `512 / NFFT` before model inference.
pub struct StftProcessor {
    nfft: usize,
    hop_size: usize,
    spectrum_scale: f32,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    full_spectrum: Vec<Complex<f32>>,
    overlap_buffer: Vec<f32>,
    fft_scratch: Vec<f32>,
    ifft_scratch: Vec<f32>,
    output_spectrum: [(f32, f32); NUM_FREQ_BINS],
    output_samples: Vec<f32>,
    /// Original full spectrum saved during `analyze()` for HF reconstruction.
    original_spectrum_saved: Vec<Complex<f32>>,
    /// High-band processor for split-band reconstruction.
    hf_processor: HighBandProcessor,
}

impl StftProcessor {
    /// Creates a processor tuned for the given host sample rate.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let ratio = sample_rate as f64 / MODEL_SR;
        let nfft_raw = (MODEL_NFFT * ratio).round() as usize;
        // Ensure even (required by realfft)
        let nfft = if nfft_raw.is_multiple_of(2) {
            nfft_raw
        } else {
            nfft_raw + 1
        };
        let hop_size = nfft / 2;
        let spectrum_scale = MODEL_NFFT as f32 / nfft as f32;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        let ifft = planner.plan_fft_inverse(nfft);
        let n_bins = nfft / 2 + 1;
        let hf_bin_count = n_bins.saturating_sub(NUM_FREQ_BINS);

        // sqrt(Hann) window — periodic style matching PyTorch
        let window: Vec<f32> = (0..nfft)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / nfft as f32;
                let hann = 0.5 * (1.0 - phase.cos());
                hann.sqrt()
            })
            .collect();

        Self {
            nfft,
            hop_size,
            spectrum_scale,
            window,
            fft,
            ifft,
            full_spectrum: vec![Complex::new(0.0, 0.0); n_bins],
            overlap_buffer: vec![0.0; nfft],
            fft_scratch: vec![0.0; nfft],
            ifft_scratch: vec![0.0; nfft],
            output_spectrum: [(0.0, 0.0); NUM_FREQ_BINS],
            output_samples: vec![0.0; hop_size],
            original_spectrum_saved: vec![Complex::new(0.0, 0.0); n_bins],
            hf_processor: HighBandProcessor::new(hf_bin_count, hop_size, sample_rate),
        }
    }

    /// Computed FFT size for the host sample rate.
    #[must_use]
    pub const fn nfft(&self) -> usize {
        self.nfft
    }

    /// Computed hop size (NFFT / 2).
    #[must_use]
    pub const fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Returns the original full spectrum saved during the last `analyze()` call.
    ///
    /// Contains all `n_bins` complex values (model bins 0..257 + HF bins 257..N).
    /// Must be copied by the caller before `synthesize()` overwrites the internal
    /// spectrum buffer.
    #[must_use]
    pub fn original_spectrum(&self) -> &[Complex<f32>] {
        &self.original_spectrum_saved
    }

    /// Analyzes a time-domain frame and returns the first 257 bins
    /// scaled by `spectrum_scale` for model compatibility.
    ///
    /// Also saves the full original spectrum (accessible via
    /// [`original_spectrum()`]) for later HF reconstruction.
    #[inline]
    pub fn analyze(&mut self, frame: &[f32]) -> &[(f32, f32); NUM_FREQ_BINS] {
        for (i, (&sample, &win)) in frame.iter().zip(&self.window).enumerate() {
            self.fft_scratch[i] = sample * win;
        }

        if self
            .fft
            .process(&mut self.fft_scratch, &mut self.full_spectrum)
            .is_err()
        {
            self.output_spectrum = [(0.0, 0.0); NUM_FREQ_BINS];
            return &self.output_spectrum;
        }

        // Save full spectrum before synthesize() overwrites it
        self.original_spectrum_saved
            .copy_from_slice(&self.full_spectrum);

        // Extract and scale only the first 257 bins for the model
        for i in 0..NUM_FREQ_BINS {
            self.output_spectrum[i] = (
                self.full_spectrum[i].re * self.spectrum_scale,
                self.full_spectrum[i].im * self.spectrum_scale,
            );
        }

        &self.output_spectrum
    }

    /// Synthesises time-domain samples from split-band processing.
    ///
    /// The low band (257 model bins) is un-scaled from the neural network
    /// output. The high band is processed by [`HighBandProcessor`]:
    /// a transient-aware spectral gate (adequate SNR) or air exciter
    /// synthesis (noisy HF). A 4-bin raised-cosine crossfade at the 8 kHz
    /// boundary ensures phase continuity.
    ///
    /// * `model_spectrum` – clean 257-bin output from the neural network.
    /// * `original_full_spectrum` – full spectrum from `analyze()` for this frame.
    /// * `vad_strength` – voice activity gate (0.0–1.0).
    /// * `transient_flag` – force transient bypass for the HF gate.
    /// * `voice_enhance` – HF reconstruction level (0.0 = cut all HF, 1.0 = full).
    #[inline]
    pub fn synthesize(
        &mut self,
        model_spectrum: &[(f32, f32); NUM_FREQ_BINS],
        original_full_spectrum: &[Complex<f32>],
        vad_strength: f32,
        transient_flag: bool,
        voice_enhance: f32,
    ) -> &[f32] {
        let inv_scale = 1.0 / self.spectrum_scale;
        let n_bins = self.nfft / 2 + 1;
        let hf_count = n_bins - NUM_FREQ_BINS;

        // ── Low band: un-scale model bins ──────────────────────────────
        for (i, &(re, im)) in model_spectrum.iter().enumerate() {
            self.full_spectrum[i] = Complex::new(re * inv_scale, im * inv_scale);
        }

        // ── High band: process via HighBandProcessor ───────────────────
        if hf_count > 0 && original_full_spectrum.len() >= n_bins {
            let hf_original = &original_full_spectrum[NUM_FREQ_BINS..n_bins];

            // Split borrow: hf_processor (&mut) + full_spectrum slice (&mut)
            self.hf_processor.process_high_band(
                hf_original,
                model_spectrum,
                vad_strength,
                transient_flag,
                &mut self.full_spectrum[NUM_FREQ_BINS..n_bins],
            );

            // ── Crossover: 4-bin raised-cosine at the boundary ─────────
            // Bins 255, 256 (model) ↔ Bins 257, 258 (HF)
            if hf_count >= CROSSFADE_BINS / 2 {
                let half = CROSSFADE_BINS / 2;

                // Snapshot pure boundary values
                let m0 = self.full_spectrum[NUM_FREQ_BINS - 2]; // bin 255
                let m1 = self.full_spectrum[NUM_FREQ_BINS - 1]; // bin 256
                let h0 = self.full_spectrum[NUM_FREQ_BINS]; // bin 257
                let h1 = if hf_count > 1 {
                    self.full_spectrum[NUM_FREQ_BINS + 1] // bin 258
                } else {
                    h0
                };

                for k in 0..CROSSFADE_BINS {
                    let t = (k as f32 + 0.5) / CROSSFADE_BINS as f32;
                    let w = 0.5 * (1.0 - (PI * t).cos()); // raised cosine

                    let model_val = if k < half { [m0, m1][k] } else { m1 };
                    let hf_val = if k >= half { [h0, h1][k - half] } else { h0 };

                    let bin = NUM_FREQ_BINS - half + k;
                    self.full_spectrum[bin] = Complex::new(
                        model_val.re * (1.0 - w) + hf_val.re * w,
                        model_val.im * (1.0 - w) + hf_val.im * w,
                    );
                }
            }
        } else {
            for i in NUM_FREQ_BINS..n_bins {
                self.full_spectrum[i] = Complex::new(0.0, 0.0);
            }
        }

        // Scale HF band by voice_enhance level.
        // At 0.0 the entire HF (>8 kHz) is silenced; at 1.0 it passes fully.
        if voice_enhance < 0.99 {
            for bin in &mut self.full_spectrum[NUM_FREQ_BINS..n_bins] {
                bin.re *= voice_enhance;
                bin.im *= voice_enhance;
            }
        }

        // DC and Nyquist must have zero imaginary part
        self.full_spectrum[0].im = 0.0;
        self.full_spectrum[n_bins - 1].im = 0.0;

        // Inverse FFT
        if self
            .ifft
            .process(&mut self.full_spectrum, &mut self.ifft_scratch)
            .is_err()
        {
            self.output_samples.fill(0.0);
            return &self.output_samples;
        }

        // Normalize and apply synthesis window
        let scale = 1.0 / self.nfft as f32;
        for (i, sample) in self.ifft_scratch.iter_mut().enumerate() {
            *sample *= scale * self.window[i];
        }

        // Overlap-add
        for (i, &sample) in self.ifft_scratch.iter().enumerate() {
            self.overlap_buffer[i] += sample;
        }

        self.output_samples
            .copy_from_slice(&self.overlap_buffer[..self.hop_size]);

        self.overlap_buffer.copy_within(self.hop_size.., 0);
        self.overlap_buffer[self.nfft - self.hop_size..].fill(0.0);

        &self.output_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfft_48khz() {
        let stft = StftProcessor::new(48000);
        // round(512 * 48000 / 16000) = round(1536) = 1536
        assert_eq!(stft.nfft(), 1536);
        assert_eq!(stft.hop_size(), 768);
    }

    #[test]
    fn nfft_16khz() {
        let stft = StftProcessor::new(16000);
        assert_eq!(stft.nfft(), 512);
        assert_eq!(stft.hop_size(), 256);
    }

    #[test]
    fn nfft_44100() {
        let stft = StftProcessor::new(44100);
        // round(512 * 44100 / 16000) = round(1411.2) = 1412 (even? 1412 is even ✓)
        let nfft = stft.nfft();
        assert!(nfft.is_multiple_of(2), "NFFT must be even, got {nfft}");
        assert_eq!(stft.hop_size(), nfft / 2);
    }

    #[test]
    fn nfft_96khz() {
        let stft = StftProcessor::new(96000);
        // round(512 * 96000 / 16000) = 3072
        assert_eq!(stft.nfft(), 3072);
    }

    #[test]
    fn nfft_always_even() {
        // Test several sample rates to ensure NFFT is always even
        for sr in [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000] {
            let stft = StftProcessor::new(sr);
            assert!(
                stft.nfft().is_multiple_of(2),
                "NFFT must be even for sr={sr}, got {}",
                stft.nfft()
            );
        }
    }

    #[test]
    fn analyze_returns_correct_size() {
        let mut stft = StftProcessor::new(48000);
        let frame = vec![0.0f32; stft.nfft()];
        let spectrum = stft.analyze(&frame);
        assert_eq!(spectrum.len(), NUM_FREQ_BINS);
    }

    #[test]
    fn analyze_silence_produces_near_zero() {
        let mut stft = StftProcessor::new(48000);
        let frame = vec![0.0f32; stft.nfft()];
        let spectrum = stft.analyze(&frame);
        let energy: f32 = spectrum.iter().map(|(re, im)| re * re + im * im).sum();
        assert!(energy < 1e-10, "silence should produce zero spectrum, energy={energy}");
    }

    #[test]
    fn original_spectrum_length() {
        let mut stft = StftProcessor::new(48000);
        let frame = vec![0.0f32; stft.nfft()];
        let _ = stft.analyze(&frame);
        let orig = stft.original_spectrum();
        assert_eq!(orig.len(), stft.nfft() / 2 + 1);
    }

    #[test]
    fn stft_round_trip_preserves_energy() {
        // Analyze → pass through unmodified → synthesize → check output has energy
        let mut stft = StftProcessor::new(48000);
        let nfft = stft.nfft();

        // Create a 440 Hz tone frame
        let frame: Vec<f32> = (0..nfft)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        let spectrum = stft.analyze(&frame);
        let spectrum_owned = *spectrum;
        let orig_full = stft.original_spectrum().to_vec();

        // Synthesize with passthrough (no model modification)
        let output = stft.synthesize(&spectrum_owned, &orig_full, 1.0, false, 1.0);

        // Output should have non-trivial energy (it's the first frame so overlap-add
        // only produces hop_size samples)
        let energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(
            energy > 0.0,
            "round-trip should produce non-zero output, got energy={energy}"
        );
    }

    #[test]
    fn synthesize_output_length() {
        let mut stft = StftProcessor::new(48000);
        let nfft = stft.nfft();
        let frame = vec![0.0f32; nfft];
        let spectrum = stft.analyze(&frame);
        let spectrum_owned = *spectrum;
        let orig_full = stft.original_spectrum().to_vec();
        let output = stft.synthesize(&spectrum_owned, &orig_full, 1.0, false, 1.0);
        assert_eq!(output.len(), stft.hop_size());
    }
}
