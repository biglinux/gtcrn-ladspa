/// Second-order IIR biquad filter (Direct Form II Transposed).
///
/// Used as a highpass pre-filter to remove sub-audible content that
/// confuses the GTCRN model.
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Create a 2nd-order Butterworth highpass filter.
    ///
    /// `cutoff_hz` — cutoff frequency in Hz
    /// `sample_rate` — sample rate in Hz
    pub fn highpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q = 1/sqrt(2) for Butterworth
        let alpha = sin_w0 / (2.0 * std::f32::consts::FRAC_1_SQRT_2);

        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;

        Self {
            b0: ((1.0 + cos_w0) / 2.0) * inv_a0,
            b1: (-(1.0 + cos_w0)) * inv_a0,
            b2: ((1.0 + cos_w0) / 2.0) * inv_a0,
            a1: (-2.0 * cos_w0) * inv_a0,
            a2: (1.0 - alpha) * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Create a 2nd-order Butterworth lowpass filter.
    ///
    /// `cutoff_hz` — cutoff frequency in Hz
    /// `sample_rate` — sample rate in Hz
    pub fn lowpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q = 1/sqrt(2) for Butterworth
        let alpha = sin_w0 / (2.0 * std::f32::consts::FRAC_1_SQRT_2);

        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;

        Self {
            b0: ((1.0 - cos_w0) / 2.0) * inv_a0,
            b1: ((1.0 - cos_w0)) * inv_a0,
            b2: ((1.0 - cos_w0) / 2.0) * inv_a0,
            a1: (-2.0 * cos_w0) * inv_a0,
            a2: (1.0 - alpha) * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process a block of samples in-place.
    #[inline]
    pub fn process(&mut self, samples: &mut [f32]) {
        for x in samples.iter_mut() {
            let input = *x;
            let output = self.b0 * input + self.z1;
            self.z1 = self.b1 * input - self.a1 * output + self.z2;
            self.z2 = self.b2 * input - self.a2 * output;
            *x = output;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highpass_coefficients_are_finite() {
        let bq = Biquad::highpass(80.0, 48000.0);
        assert!(bq.b0.is_finite());
        assert!(bq.b1.is_finite());
        assert!(bq.b2.is_finite());
        assert!(bq.a1.is_finite());
        assert!(bq.a2.is_finite());
    }

    #[test]
    fn highpass_attenuates_dc() {
        let mut bq = Biquad::highpass(80.0, 48000.0);
        // Feed DC signal (constant 1.0) for 2000 samples — output should converge to ~0
        let mut samples = vec![1.0f32; 2000];
        bq.process(&mut samples);
        let last = samples.last().unwrap().abs();
        assert!(last < 0.01, "DC should be attenuated, got {last}");
    }

    #[test]
    fn highpass_passes_high_frequency() {
        let mut bq = Biquad::highpass(80.0, 48000.0);
        let sr = 48000.0;
        let freq = 1000.0; // 1 kHz — well above cutoff
        let mut samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        bq.process(&mut samples);
        // Check energy of last 480 samples (steady state)
        let energy: f32 = samples[4320..].iter().map(|x| x * x).sum::<f32>() / 480.0;
        // Input RMS² of sine = 0.5 — should pass through with minimal attenuation
        assert!(
            energy > 0.3,
            "1 kHz should pass through highpass 80 Hz, energy={energy}"
        );
    }

    #[test]
    fn highpass_attenuates_low_frequency() {
        let mut bq = Biquad::highpass(80.0, 48000.0);
        let sr = 48000.0;
        let freq = 20.0; // 20 Hz — well below cutoff
        let mut samples: Vec<f32> = (0..9600)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        bq.process(&mut samples);
        // Check energy of last 2400 samples (steady state)
        let energy: f32 = samples[7200..].iter().map(|x| x * x).sum::<f32>() / 2400.0;
        // 20 Hz is ~2 octaves below 80 Hz cutoff → ~-12 dB for 2nd order → energy < 0.05
        assert!(
            energy < 0.1,
            "20 Hz should be attenuated by 80 Hz highpass, energy={energy}"
        );
    }

    #[test]
    fn process_empty_slice() {
        let mut bq = Biquad::highpass(80.0, 48000.0);
        let mut empty: Vec<f32> = vec![];
        bq.process(&mut empty); // should not panic
    }
}
