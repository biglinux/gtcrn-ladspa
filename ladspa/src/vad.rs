//! Voice Activity Detection (VAD) helpers used by [`crate::plugin`].
//!
//! Two detectors live here so the main process loop in `plugin.rs` stays
//! readable and so each detector can be unit-tested in isolation:
//!
//! - [`dns3_energy_ratio_vad`] compares output energy to input energy after
//!   the DNS3 model has run. DNS3 preserves speech energy well, so the
//!   ratio is a strong VAD signal.
//! - [`vctk_input_only_vad`] compares input energy to a tracked noise
//!   floor. Used when only VCTK is loaded, because VCTK's aggressive
//!   suppression would otherwise drag the output-ratio VAD into a
//!   feedback loop.
//!
//! Pure functions — no allocation, no I/O, no hidden state. Callers own
//! the smoothed energy and noise-floor variables and pass them in.

use crate::model::NUM_FREQ_BINS;

/// EMA attack coefficient applied to the energy ratio in DNS3-VAD.
/// Higher = faster lock-on, lower = more stable across word boundaries.
pub const VAD_EMA_ATTACK: f32 = 0.15;

/// Energy-ratio threshold above which DNS3-VAD returns "speech".
pub const VAD_DNS3_THRESHOLD: f32 = 0.05;

/// Input-vs-noise-floor SNR threshold for VCTK-VAD (linear, ratio of energies).
pub const VAD_VCTK_SNR_THRESHOLD: f32 = 3.0;

/// Onset multiplier — input energy above `noise_floor * ONSET_MULTIPLIER`
/// counts as a speech onset, force-priming the VAD gate.
pub const ONSET_MULTIPLIER: f32 = 2.5;

/// Decay multiplier applied to the VAD gate when no speech is detected.
/// `0.95` ≈ 310 ms time constant at ~62 fps; long enough to bridge
/// micro-pauses inside a sentence.
pub const VAD_DECAY_COEFF: f32 = 0.95;

/// EMA coefficient applied to the input-energy noise floor during silence.
pub const NOISE_FLOOR_EMA: f32 = 0.05;

/// VAD-gate value below which we consider the frame "confirmed silence"
/// and let the noise-floor tracker update.
pub const NOISE_FLOOR_QUIET_THRESHOLD: f32 = 0.1;

/// VAD-gate floor below which we snap to zero, so the EMA doesn't sit at
/// numerically-tiny but non-zero values forever.
pub const VAD_GATE_FLOOR: f32 = 0.01;

/// Computes the sum of squared magnitudes for a 257-bin spectrum slice
/// represented as `(re, im)` pairs.
#[inline]
#[must_use]
pub fn spectrum_energy(spectrum: &[(f32, f32); NUM_FREQ_BINS]) -> f32 {
    spectrum.iter().map(|&(re, im)| re * re + im * im).sum()
}

/// DNS3 energy-ratio VAD. Returns `(is_speech, updated_ema)`.
///
/// The caller passes the previous EMA so this stays a pure function.
#[must_use]
pub fn dns3_energy_ratio_vad(
    input_energy: f32,
    output_energy: f32,
    previous_ema: f32,
) -> (bool, f32) {
    let energy_ratio = if input_energy > 1e-10 {
        output_energy / input_energy
    } else {
        0.0
    };
    let updated_ema = previous_ema * (1.0 - VAD_EMA_ATTACK) + energy_ratio * VAD_EMA_ATTACK;
    (updated_ema > VAD_DNS3_THRESHOLD, updated_ema)
}

/// VCTK input-only VAD. Returns `true` when input energy is above the
/// running noise-floor estimate by at least the SNR threshold.
#[must_use]
pub fn vctk_input_only_vad(input_energy: f32, noise_floor: f32) -> bool {
    noise_floor > 1e-10 && input_energy > noise_floor * VAD_VCTK_SNR_THRESHOLD
}

/// Onset detector. Returns `true` when input energy exceeds
/// `noise_floor * ONSET_MULTIPLIER` — this catches the first soft
/// syllable before the per-frame VAD has had time to ramp up.
#[must_use]
pub fn detect_onset(input_energy: f32, noise_floor: f32) -> bool {
    noise_floor > 1e-10 && input_energy > noise_floor * ONSET_MULTIPLIER
}

/// Update the running noise-floor EMA when the frame is confirmed silence.
///
/// The first non-zero update bootstraps from `input_energy` to skip the
/// long warm-up that a pure EMA would otherwise need.
pub fn update_noise_floor(
    noise_floor: &mut f32,
    input_energy: f32,
    vad_gate: f32,
    is_speech: bool,
) {
    if vad_gate >= NOISE_FLOOR_QUIET_THRESHOLD || is_speech {
        return;
    }
    if *noise_floor < 1e-10 {
        *noise_floor = input_energy;
    } else {
        *noise_floor = *noise_floor * (1.0 - NOISE_FLOOR_EMA) + input_energy * NOISE_FLOOR_EMA;
    }
}

/// Mutates `vad_gate` and `vad_energy_smooth` to reflect this frame's
/// detection result. On a fresh onset we prime the energy EMA so the
/// downstream gate threshold opens on the first speech sample instead of
/// waiting for the EMA to ramp.
pub fn step_vad_gate(
    vad_gate: &mut f32,
    vad_energy_smooth: &mut f32,
    is_speech: bool,
    future_speech: bool,
    onset: bool,
) {
    if is_speech || future_speech || onset {
        *vad_gate = 1.0;
        if onset && *vad_energy_smooth < 0.2 {
            *vad_energy_smooth = 0.2;
        }
    } else {
        *vad_gate *= VAD_DECAY_COEFF;
        if *vad_gate < VAD_GATE_FLOOR {
            *vad_gate = 0.0;
        }
    }
}

/// Linearly interpolate between the silence and speech strengths driven
/// by the current VAD gate value (`0.0`=silence, `1.0`=speech).
#[inline]
#[must_use]
pub fn effective_strength(noise_strength: f32, speech_strength: f32, vad_gate: f32) -> f32 {
    if vad_gate > 0.0 {
        noise_strength * (1.0 - vad_gate) + speech_strength * vad_gate
    } else {
        noise_strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dns3_vad_speech_when_ratio_high() {
        let (is_speech, ema) = dns3_energy_ratio_vad(1.0, 0.5, 0.0);
        // First call: ema = 0.5 * 0.15 = 0.075, above 0.05 threshold
        assert!(is_speech);
        assert!((ema - 0.075).abs() < 1e-6);
    }

    #[test]
    fn dns3_vad_silence_when_ratio_zero() {
        let (is_speech, ema) = dns3_energy_ratio_vad(0.0, 0.0, 0.0);
        assert!(!is_speech);
        assert!(ema.abs() < 1e-6);
    }

    #[test]
    fn vctk_vad_below_threshold() {
        assert!(!vctk_input_only_vad(2.5, 1.0)); // 2.5 < 3.0 * 1.0
        assert!(vctk_input_only_vad(3.5, 1.0)); // 3.5 > 3.0 * 1.0
    }

    #[test]
    fn vctk_vad_uninitialised_floor() {
        // Floor at exactly 0 means tracker hasn't seen a silence frame
        // yet — refuse to call anything speech.
        assert!(!vctk_input_only_vad(10.0, 0.0));
    }

    #[test]
    fn onset_detected_when_input_jumps() {
        assert!(detect_onset(3.0, 1.0)); // 3.0 > 1.0 * 2.5
        assert!(!detect_onset(2.0, 1.0)); // 2.0 < 1.0 * 2.5
    }

    #[test]
    fn noise_floor_updates_during_silence() {
        let mut floor = 0.0;
        update_noise_floor(&mut floor, 0.5, 0.0, false);
        assert!((floor - 0.5).abs() < 1e-6); // bootstrap

        update_noise_floor(&mut floor, 1.0, 0.0, false);
        let expected = 0.5 * (1.0 - NOISE_FLOOR_EMA) + 1.0 * NOISE_FLOOR_EMA;
        assert!((floor - expected).abs() < 1e-6);
    }

    #[test]
    fn noise_floor_skips_during_speech() {
        let mut floor = 0.5;
        update_noise_floor(&mut floor, 100.0, 0.0, true); // speech: skip
        assert!((floor - 0.5).abs() < 1e-6);
    }

    #[test]
    fn vad_gate_opens_on_speech() {
        let mut gate = 0.0;
        let mut ema = 0.0;
        step_vad_gate(&mut gate, &mut ema, true, false, false);
        assert!((gate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vad_gate_decays_on_silence() {
        let mut gate = 1.0;
        let mut ema = 0.5;
        step_vad_gate(&mut gate, &mut ema, false, false, false);
        assert!((gate - VAD_DECAY_COEFF).abs() < 1e-6);
    }

    #[test]
    fn vad_gate_snaps_to_zero_below_floor() {
        let mut gate = 0.005; // below VAD_GATE_FLOOR = 0.01
        let mut ema = 0.0;
        step_vad_gate(&mut gate, &mut ema, false, false, false);
        assert!(gate.abs() < 1e-6);
    }

    #[test]
    fn onset_primes_energy_ema() {
        let mut gate = 0.0;
        let mut ema = 0.05;
        step_vad_gate(&mut gate, &mut ema, false, false, true);
        assert!((gate - 1.0).abs() < 1e-6);
        assert!((ema - 0.2).abs() < 1e-6);
    }

    #[test]
    fn effective_strength_silence() {
        assert!((effective_strength(0.5, 1.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn effective_strength_full_speech() {
        assert!((effective_strength(0.5, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn effective_strength_midway() {
        let v = effective_strength(0.0, 1.0, 0.5);
        assert!((v - 0.5).abs() < 1e-6);
    }
}
