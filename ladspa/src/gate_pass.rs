//! Gate-pass stage that runs *after* the GTCRN model on the same buffer.
//!
//! Lives in its own module because the `run()` function in `crate::plugin`
//! was carrying ~80 lines of gate state — splitting it out keeps the main
//! audio loop under the project's CCN ≤ 25 budget and lets the gate be
//! tested in isolation against the noise-reduction state.

use ladspa::PortConnection;

use crate::biquad::Biquad;
use crate::gate::{db_to_linear, smooth_coeff};
use crate::{
    PORT_GATE_ATTACK, PORT_GATE_HF_KEY, PORT_GATE_HOLD, PORT_GATE_LF_KEY, PORT_GATE_RANGE,
    PORT_GATE_RELEASE, PORT_GATE_THRESHOLD,
};

/// Sidechain envelope detector attack time (s) — fast to catch onsets.
const DET_ATTACK_S: f32 = 0.001;
/// Sidechain envelope detector release time (s) — smooth between syllables.
const DET_RELEASE_S: f32 = 0.020;

/// Bias the gate toward "open" once both energy and the spectral VAD
/// agree. `0.08` lets the gate open early on faint onsets while the
/// onset-detector primer guarantees `vad_gate` is already high when real
/// speech begins.
const VAD_GATE_OPEN_THRESHOLD: f32 = 0.08;

/// All gate-related state lives in this struct, owned by the plugin and
/// passed to [`apply_gate_pass`] each `run()` call.
pub struct GateState {
    pub hp1: Biquad,
    pub hp2: Biquad,
    pub lp1: Biquad,
    pub lp2: Biquad,
    pub last_lf_hz: f32,
    pub last_hf_hz: f32,
    pub envelope: f32,
    pub gain: f32,
    pub hold_counter: u32,
    pub is_open: bool,
    pub sidechain_buf: Vec<f32>,
}

impl GateState {
    #[must_use]
    pub fn new(sample_rate: f32, max_block: usize) -> Self {
        Self {
            hp1: Biquad::highpass(200.0, sample_rate),
            hp2: Biquad::highpass(200.0, sample_rate),
            lp1: Biquad::lowpass(4000.0, sample_rate),
            lp2: Biquad::lowpass(4000.0, sample_rate),
            last_lf_hz: 200.0,
            last_hf_hz: 4000.0,
            envelope: 0.0,
            gain: 0.0,
            hold_counter: 0,
            is_open: false,
            sidechain_buf: vec![0.0; max_block],
        }
    }

    fn refresh_filters(&mut self, lf: f32, hf: f32, sample_rate: f32) {
        if (lf - self.last_lf_hz).abs() > 0.5 {
            self.hp1 = Biquad::highpass(lf, sample_rate);
            self.hp2 = Biquad::highpass(lf, sample_rate);
            self.last_lf_hz = lf;
        }
        if (hf - self.last_hf_hz).abs() > 0.5 {
            self.lp1 = Biquad::lowpass(hf, sample_rate);
            self.lp2 = Biquad::lowpass(hf, sample_rate);
            self.last_hf_hz = hf;
        }
    }

    fn fill_sidechain(&mut self, input: &[f32], sample_count: usize) {
        if self.sidechain_buf.len() < sample_count {
            self.sidechain_buf.resize(sample_count, 0.0);
        }
        self.sidechain_buf[..sample_count].copy_from_slice(&input[..sample_count]);
        // 4th-order bandpass: HP(LF)×2 → LP(HF)×2
        self.hp1.process(&mut self.sidechain_buf[..sample_count]);
        self.hp2.process(&mut self.sidechain_buf[..sample_count]);
        self.lp1.process(&mut self.sidechain_buf[..sample_count]);
        self.lp2.process(&mut self.sidechain_buf[..sample_count]);
    }
}

/// Per-call gate parameters resolved from LADSPA control ports. Built by
/// [`read_gate_params`]; passed through `apply_gate_pass` so the inner
/// per-sample loop has constants in registers instead of `*ports[…]`.
struct GateParams {
    threshold_linear: f32,
    range_linear: f32,
    attack_coeff: f32,
    release_coeff: f32,
    hold_samples: u32,
    det_attack: f32,
    det_release: f32,
}

fn read_gate_params<'a>(ports: &[&'a PortConnection<'a>], sample_rate: f32) -> GateParams {
    let threshold_db = *ports[PORT_GATE_THRESHOLD].unwrap_control();
    let attack_ms = *ports[PORT_GATE_ATTACK].unwrap_control();
    let hold_ms = *ports[PORT_GATE_HOLD].unwrap_control();
    let release_ms = *ports[PORT_GATE_RELEASE].unwrap_control();
    let range_db = *ports[PORT_GATE_RANGE].unwrap_control();

    GateParams {
        threshold_linear: db_to_linear(threshold_db.clamp(-80.0, 0.0)),
        range_linear: db_to_linear(range_db.clamp(-90.0, 0.0)),
        attack_coeff: smooth_coeff(attack_ms.max(0.1) / 1000.0, sample_rate),
        release_coeff: smooth_coeff(release_ms.max(1.0) / 1000.0, sample_rate),
        hold_samples: ((hold_ms.max(0.0) / 1000.0) * sample_rate) as u32,
        det_attack: smooth_coeff(DET_ATTACK_S, sample_rate),
        det_release: smooth_coeff(DET_RELEASE_S, sample_rate),
    }
}

fn step_envelope(envelope: &mut f32, sc_level: f32, det_attack: f32, det_release: f32) {
    if sc_level > *envelope {
        *envelope += det_attack * (sc_level - *envelope);
    } else {
        *envelope += det_release * (sc_level - *envelope);
    }
}

fn step_gate_state(state: &mut GateState, params: &GateParams, vad_gate: f32) {
    if state.envelope >= params.threshold_linear && vad_gate > VAD_GATE_OPEN_THRESHOLD {
        state.is_open = true;
        state.hold_counter = params.hold_samples;
    } else if state.hold_counter > 0 {
        state.hold_counter -= 1;
    } else {
        state.is_open = false;
    }
}

fn step_gain_smoothing(state: &mut GateState, params: &GateParams) {
    let target = if state.is_open {
        1.0
    } else {
        params.range_linear
    };
    if state.gain < target {
        state.gain += params.attack_coeff * (target - state.gain);
    } else {
        state.gain += params.release_coeff * (target - state.gain);
    }
}

/// Apply the gate pass to `output` in-place. `input` is the raw mic
/// buffer (pre-GTCRN) used for sidechain detection — feeding the raw
/// signal opens the gate before GTCRN's latency would otherwise clip
/// the first syllable. `vad_gate` comes from the spectral VAD upstream.
pub fn apply_gate_pass<'a>(
    state: &mut GateState,
    sample_rate: f32,
    vad_gate: f32,
    input: &[f32],
    output: &mut [f32],
    sample_count: usize,
    ports: &[&'a PortConnection<'a>],
) {
    let threshold_db = *ports[PORT_GATE_THRESHOLD].unwrap_control();
    // Bypass when threshold is at the +0 dB ceiling.
    if threshold_db >= 0.0 {
        return;
    }

    let lf = ports[PORT_GATE_LF_KEY]
        .unwrap_control()
        .clamp(20.0, 20000.0);
    let hf = ports[PORT_GATE_HF_KEY]
        .unwrap_control()
        .clamp(20.0, 20000.0);
    state.refresh_filters(lf, hf, sample_rate);

    let params = read_gate_params(ports, sample_rate);
    state.fill_sidechain(input, sample_count);

    for (i, sample) in output.iter_mut().enumerate().take(sample_count) {
        let sc_level = state.sidechain_buf[i].abs();
        step_envelope(
            &mut state.envelope,
            sc_level,
            params.det_attack,
            params.det_release,
        );
        step_gate_state(state, &params, vad_gate);
        step_gain_smoothing(state, &params);
        *sample *= state.gain;
    }
}
