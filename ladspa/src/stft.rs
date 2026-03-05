//! Real-time STFT/iSTFT processing with overlap-add synthesis.
//!
//! Adapts the FFT size to the host sample rate so that the first 257 bins
//! always cover 0–8 kHz (matching the 16 kHz model's frequency range).
//! The `sqrt(hann)` window ensures perfect reconstruction with 50% overlap.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::model::NUM_FREQ_BINS;

/// FFT size the GTCRN model was trained with (at 16 kHz).
const MODEL_NFFT: f64 = 512.0;

/// Sample rate the model was trained at.
const MODEL_SR: f64 = 16_000.0;

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

    /// Analyzes a time-domain frame and returns the first 257 bins
    /// scaled by `spectrum_scale` for model compatibility.
    #[inline]
    pub fn analyze(&mut self, frame: &[f32]) -> &[(f32, f32); NUM_FREQ_BINS] {
        for (i, (&sample, &win)) in frame.iter().zip(&self.window).enumerate() {
            self.fft_scratch[i] = sample * win;
        }

        self.fft
            .process(&mut self.fft_scratch, &mut self.full_spectrum)
            .expect("FFT processing failed");

        // Extract and scale only the first 257 bins
        for i in 0..NUM_FREQ_BINS {
            self.output_spectrum[i] = (
                self.full_spectrum[i].re * self.spectrum_scale,
                self.full_spectrum[i].im * self.spectrum_scale,
            );
        }

        &self.output_spectrum
    }

    /// Synthesizes time-domain samples from a 257-bin model spectrum.
    ///
    /// The model bins are un-scaled back into the first 257 positions
    /// of the full spectrum; remaining bins are zeroed.
    #[inline]
    pub fn synthesize(&mut self, spectrum: &[(f32, f32); NUM_FREQ_BINS]) -> &[f32] {
        let inv_scale = 1.0 / self.spectrum_scale;

        // First 257 bins: un-scale from model
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.full_spectrum[i] = Complex::new(re * inv_scale, im * inv_scale);
        }

        // Remaining bins (257..769): zero
        let n_bins = self.nfft / 2 + 1;
        for i in NUM_FREQ_BINS..n_bins {
            self.full_spectrum[i] = Complex::new(0.0, 0.0);
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
