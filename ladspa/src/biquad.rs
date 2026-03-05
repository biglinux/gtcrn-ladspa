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
