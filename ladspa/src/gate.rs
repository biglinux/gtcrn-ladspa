//! LADSPA Noise Gate with sidechain bandpass key filter.
//!
//! A speech-optimized gate that uses highpass + lowpass filters on the
//! sidechain to focus detection on the 200–4000 Hz speech band, rejecting
//! impulsive noises (mouse clicks, key taps) that fall outside this range.
//!
//! Unlike swh gate_1410 whose key filter controls are broken, this
//! implementation handles filter frequencies directly in Hz — no
//! `HINT_SAMPLE_RATE` normalization issues.

use crate::biquad::Biquad;
use ladspa::{Plugin, PluginDescriptor, PortConnection};

// =============================================================================
// Port Indices
// =============================================================================

const PORT_INPUT: usize = 0;
const PORT_OUTPUT: usize = 1;
const PORT_THRESHOLD: usize = 2;
const PORT_ATTACK: usize = 3;
const PORT_HOLD: usize = 4;
const PORT_RELEASE: usize = 5;
const PORT_RANGE: usize = 6;
const PORT_LF_KEY: usize = 7;
const PORT_HF_KEY: usize = 8;

// =============================================================================
// Detector Constants
// =============================================================================

/// Envelope detector attack time in seconds (fast, to catch speech onsets).
const DET_ATTACK_S: f32 = 0.001;

/// Envelope detector release time in seconds (smooth between syllables).
const DET_RELEASE_S: f32 = 0.020;

// =============================================================================
// Plugin
// =============================================================================

/// Noise gate with sidechain bandpass key filter.
///
/// Uses cascaded (4th-order) Butterworth filters for the sidechain bandpass,
/// providing -24 dB/octave rejection of out-of-band signals.
pub struct GatePlugin {
    sample_rate: f32,

    // Sidechain bandpass filters (cascaded for 4th-order / -24 dB/oct)
    hp_filter_1: Biquad,
    hp_filter_2: Biquad,
    lp_filter_1: Biquad,
    lp_filter_2: Biquad,
    last_lf_hz: f32,
    last_hf_hz: f32,

    // State
    envelope: f32,
    gate_gain: f32,
    hold_counter: u32,
    is_open: bool,

    // Pre-allocated sidechain buffer
    sidechain_buf: Vec<f32>,
}

impl GatePlugin {
    #[must_use]
    pub fn new(_desc: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
        let sr = sample_rate as f32;
        let default_lf = 200.0;
        let default_hf = 4000.0;

        Box::new(Self {
            sample_rate: sr,
            hp_filter_1: Biquad::highpass(default_lf, sr),
            hp_filter_2: Biquad::highpass(default_lf, sr),
            lp_filter_1: Biquad::lowpass(default_hf, sr),
            lp_filter_2: Biquad::lowpass(default_hf, sr),
            last_lf_hz: default_lf,
            last_hf_hz: default_hf,
            envelope: 0.0,
            gate_gain: 0.0, // Start closed
            hold_counter: 0,
            is_open: false,
            sidechain_buf: Vec::with_capacity(1024),
        })
    }
}

impl Plugin for GatePlugin {
    fn activate(&mut self) {}

    fn deactivate(&mut self) {}

    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input = ports[PORT_INPUT].unwrap_audio();
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut();

        let threshold_db = *ports[PORT_THRESHOLD].unwrap_control();
        let attack_ms = *ports[PORT_ATTACK].unwrap_control();
        let hold_ms = *ports[PORT_HOLD].unwrap_control();
        let release_ms = *ports[PORT_RELEASE].unwrap_control();
        let range_db = *ports[PORT_RANGE].unwrap_control();
        let lf_hz = *ports[PORT_LF_KEY].unwrap_control();
        let hf_hz = *ports[PORT_HF_KEY].unwrap_control();

        // Recreate sidechain filters if frequency controls changed
        let lf_clamped = lf_hz.clamp(20.0, 20000.0);
        let hf_clamped = hf_hz.clamp(20.0, 20000.0);

        if (lf_clamped - self.last_lf_hz).abs() > 0.5 {
            self.hp_filter_1 = Biquad::highpass(lf_clamped, self.sample_rate);
            self.hp_filter_2 = Biquad::highpass(lf_clamped, self.sample_rate);
            self.last_lf_hz = lf_clamped;
        }
        if (hf_clamped - self.last_hf_hz).abs() > 0.5 {
            self.lp_filter_1 = Biquad::lowpass(hf_clamped, self.sample_rate);
            self.lp_filter_2 = Biquad::lowpass(hf_clamped, self.sample_rate);
            self.last_hf_hz = hf_clamped;
        }

        // Pre-compute constants
        let threshold_linear = db_to_linear(threshold_db.clamp(-80.0, 0.0));
        let range_linear = db_to_linear(range_db.clamp(-90.0, 0.0));
        let gate_attack = smooth_coeff(attack_ms.max(0.1) / 1000.0, self.sample_rate);
        let gate_release = smooth_coeff(release_ms.max(1.0) / 1000.0, self.sample_rate);
        let hold_samples = ((hold_ms.max(0.0) / 1000.0) * self.sample_rate) as u32;

        let det_attack = smooth_coeff(DET_ATTACK_S, self.sample_rate);
        let det_release = smooth_coeff(DET_RELEASE_S, self.sample_rate);

        // Prepare sidechain buffer (reuse allocation)
        self.sidechain_buf.resize(sample_count, 0.0);
        self.sidechain_buf[..sample_count].copy_from_slice(&input[..sample_count]);

        // Apply 4th-order bandpass to sidechain: HP(LF) × 2 → LP(HF) × 2
        self.hp_filter_1.process(&mut self.sidechain_buf[..sample_count]);
        self.hp_filter_2.process(&mut self.sidechain_buf[..sample_count]);
        self.lp_filter_1.process(&mut self.sidechain_buf[..sample_count]);
        self.lp_filter_2.process(&mut self.sidechain_buf[..sample_count]);

        // Per-sample processing
        for i in 0..sample_count {
            let sc_level = self.sidechain_buf[i].abs();

            // Peak envelope follower
            if sc_level > self.envelope {
                self.envelope += det_attack * (sc_level - self.envelope);
            } else {
                self.envelope += det_release * (sc_level - self.envelope);
            }

            // Gate state machine
            if self.envelope >= threshold_linear {
                self.is_open = true;
                self.hold_counter = hold_samples;
            } else if self.hold_counter > 0 {
                self.hold_counter -= 1;
                // Stay open during hold period
            } else {
                self.is_open = false;
            }

            // Smooth gate gain toward target
            let target = if self.is_open { 1.0 } else { range_linear };
            if self.gate_gain < target {
                self.gate_gain += gate_attack * (target - self.gate_gain);
            } else {
                self.gate_gain += gate_release * (target - self.gate_gain);
            }

            output[i] = input[i] * self.gate_gain;
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert dB to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Compute one-pole smoothing coefficient from time constant.
///
/// Returns `1 - exp(-1 / (time_s * sample_rate))`.
/// Higher values = faster response.
#[inline]
fn smooth_coeff(time_s: f32, sample_rate: f32) -> f32 {
    if time_s <= 0.0 {
        return 1.0;
    }
    1.0 - (-1.0 / (time_s * sample_rate)).exp()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_to_linear_conversions() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-6);
        assert!((db_to_linear(-90.0)).abs() < 1e-4);
    }

    #[test]
    fn smooth_coeff_ranges() {
        let c = smooth_coeff(0.01, 48000.0);
        assert!(c > 0.0 && c < 1.0, "coeff={c}");
        // Faster time → larger coefficient
        let c_fast = smooth_coeff(0.001, 48000.0);
        assert!(c_fast > c, "faster time should give larger coeff");
    }

    #[test]
    fn gate_closes_on_silence() {
        // Create gate at 48kHz
        let desc = test_descriptor();
        let mut gate = GatePlugin::new(&desc, 48000);

        // Feed 4800 samples of silence (100ms) — gate should stay closed
        let silence = vec![0.0f32; 4800];
        let mut out = vec![0.0f32; 4800];
        let threshold = -30.0f32;
        let attack = 10.0f32;
        let hold = 400.0f32;
        let release = 200.0f32;
        let range = -90.0f32;
        let lf = 200.0f32;
        let hf = 4000.0f32;

        let ports = make_ports(&silence, &mut out, &threshold, &attack, &hold, &release, &range, &lf, &hf);
        let port_refs: Vec<&PortConnection> = ports.iter().collect();
        gate.run(4800, &port_refs);

        // Output should be near-silent (attenuated by range_db)
        let rms: f32 = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms < 0.001, "silence should stay silent, rms={rms}");
    }

    #[test]
    fn gate_opens_on_speech_band_signal() {
        let desc = test_descriptor();
        let mut gate = GatePlugin::new(&desc, 48000);

        let sr = 48000.0;
        let freq = 1000.0; // 1kHz — within 200-4000 Hz band
        let amp = 0.5; // -6 dBFS, well above -30 dB threshold

        // 200ms of 1kHz tone
        let n = 9600;
        let tone: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        let mut out = vec![0.0f32; n];

        let threshold = -30.0f32;
        let attack = 10.0f32;
        let hold = 400.0f32;
        let release = 200.0f32;
        let range = -90.0f32;
        let lf = 200.0f32;
        let hf = 4000.0f32;

        let ports = make_ports(&tone, &mut out, &threshold, &attack, &hold, &release, &range, &lf, &hf);
        let port_refs: Vec<&PortConnection> = ports.iter().collect();
        gate.run(n, &port_refs);

        // Steady-state output (last 50ms) should pass through with high energy
        let tail = &out[n - 2400..];
        let rms: f32 = (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms > 0.2, "1kHz tone should open gate, rms={rms}");
    }

    #[test]
    fn key_filter_rejects_below_lf() {
        let desc = test_descriptor();
        let mut gate = GatePlugin::new(&desc, 48000);

        let sr = 48000.0;
        let freq = 50.0; // 50 Hz — below LF key filter of 200 Hz
        let amp = 0.5;

        let n = 48000; // 1 second
        let tone: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        let mut out = vec![0.0f32; n];

        let threshold = -30.0f32;
        let attack = 10.0f32;
        let hold = 0.0f32; // No hold — gate should close quickly
        let release = 10.0f32; // Fast release
        let range = -90.0f32;
        let lf = 200.0f32;
        let hf = 4000.0f32;

        let ports = make_ports(&tone, &mut out, &threshold, &attack, &hold, &release, &range, &lf, &hf);
        let port_refs: Vec<&PortConnection> = ports.iter().collect();
        gate.run(n, &port_refs);

        // 50 Hz is heavily attenuated by the 200 Hz highpass key filter.
        // The gate should NOT open (or barely open), so output should be very quiet.
        let tail = &out[n / 2..]; // Second half (steady state)
        let rms: f32 = (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms < 0.05, "50 Hz should NOT open gate with LF=200 Hz, rms={rms}");
    }

    #[test]
    fn key_filter_rejects_above_hf() {
        let desc = test_descriptor();
        let mut gate = GatePlugin::new(&desc, 48000);

        let sr = 48000.0;
        let freq = 10000.0; // 10 kHz — above HF key filter of 4000 Hz
        let amp = 0.5;

        let n = 48000; // 1 second
        let tone: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        let mut out = vec![0.0f32; n];

        let threshold = -30.0f32;
        let attack = 10.0f32;
        let hold = 0.0f32;
        let release = 10.0f32;
        let range = -90.0f32;
        let lf = 200.0f32;
        let hf = 4000.0f32;

        let ports = make_ports(&tone, &mut out, &threshold, &attack, &hold, &release, &range, &lf, &hf);
        let port_refs: Vec<&PortConnection> = ports.iter().collect();
        gate.run(n, &port_refs);

        // 10 kHz is heavily attenuated by the 4000 Hz lowpass key filter.
        let tail = &out[n / 2..];
        let rms: f32 = (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms < 0.05, "10 kHz should NOT open gate with HF=4000 Hz, rms={rms}");
    }

    // =========================================================================
    // Test helpers
    // =========================================================================

    fn test_descriptor() -> PluginDescriptor {
        crate::gate_descriptor()
    }

    fn make_ports<'a>(
        input: &'a [f32],
        output: &'a mut [f32],
        threshold: &'a f32,
        attack: &'a f32,
        hold: &'a f32,
        release: &'a f32,
        range: &'a f32,
        lf: &'a f32,
        hf: &'a f32,
    ) -> Vec<PortConnection<'a>> {
        vec![
            PortConnection::AudioInput(input),
            PortConnection::AudioOutput(output),
            PortConnection::ControlInput(threshold),
            PortConnection::ControlInput(attack),
            PortConnection::ControlInput(hold),
            PortConnection::ControlInput(release),
            PortConnection::ControlInput(range),
            PortConnection::ControlInput(lf),
            PortConnection::ControlInput(hf),
        ]
    }
}
