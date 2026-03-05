//! GTCRN neural network model wrapper using ONNX Runtime.
//!
//! Each LADSPA instance owns its own ONNX session. On creation the model
//! is pre-warmed with a few synthetic spectrum frames so that the
//! recurrent state is away from the over-suppressive all-zeros init.

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

// =============================================================================
// Embedded Models
// =============================================================================

static EMBEDDED_MODEL_DNS3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn_simple.ort"));
static EMBEDDED_MODEL_VCTK: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn_vctk.ort"));

// =============================================================================
// Constants
// =============================================================================

pub const NUM_FREQ_BINS: usize = 257;

const CONV_SIZE: usize = 16896;
const TRA_SIZE: usize = 96;
const INTER_SIZE: usize = 1056;
const INPUT_SIZE: usize = NUM_FREQ_BINS * 2;

/// Number of synthetic frames fed to the model on creation.
/// Brings recurrent state away from zeros, avoiding cold-start suppression.
/// 20 × ~1 ms ≈ 20 ms one-time cost.
const WARMUP_FRAMES: usize = 20;

/// Factor blended towards warm-up reference on onset detection.
/// 0.0 = no reset, 1.0 = full reset to warm-up reference.
const ONSET_BLEND: f32 = 1.0;

/// EMA smoothing factor for input energy tracker (~330 ms time constant).
const ENERGY_SMOOTH: f32 = 0.05;

/// Number of frames to suppress re-triggering after an onset (~0.5 seconds).
const ONSET_COOLDOWN_FRAMES: usize = 30;

/// Energy ratio below which a frame is considered "over-suppressed".
/// 0.002 ≈ -27 dB.  DNS3 normal NR is ~-28 dB (ratio ≈ 0.0014),
/// so the flag stays armed during active noise suppression.
/// Combined with the input energy factor check, only speech-level
/// signals actually trigger the reset.
const SUPPRESS_TRIGGER_RATIO: f32 = 0.002;

/// Multiplier over energy_avg that input must exceed for the
/// suppression trigger. 6× ensures only speech-level signals trigger
/// (well above noise transient variations).
const SUPPRESS_INPUT_FACTOR: f32 = 6.0;

/// Minimum number of frames processed before onset detection activates.
/// Prevents false triggers during the first seconds when energy_avg is
/// still calibrating from near-zero warm-up values.
/// 300 frames × 16 ms ≈ 5 seconds.
const ONSET_MIN_FRAMES: usize = 300;

/// Model type selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    Dns3 = 0,
    Vctk = 1,
}

impl ModelType {
    #[must_use]
    pub fn from_control(value: f32) -> Self {
        if value >= 0.5 {
            Self::Vctk
        } else {
            Self::Dns3
        }
    }
}

// =============================================================================
// Session creation
// =============================================================================

fn create_session(model_bytes: &[u8]) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_memory(model_bytes)
}

// =============================================================================
// State
// =============================================================================

pub struct GtcrnState {
    conv: Vec<f32>,
    tra: Vec<f32>,
    inter: Vec<f32>,
    input_buf: Vec<f32>,
    output_buf: [(f32, f32); NUM_FREQ_BINS],
    /// Reference state captured after warm-up. The decay blends towards this.
    ref_conv: Vec<f32>,
    ref_tra: Vec<f32>,
    ref_inter: Vec<f32>,
}

impl GtcrnState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            conv: vec![0.0_f32; CONV_SIZE],
            tra: vec![0.0_f32; TRA_SIZE],
            inter: vec![0.0_f32; INTER_SIZE],
            input_buf: vec![0.0_f32; INPUT_SIZE],
            output_buf: [(0.0_f32, 0.0_f32); NUM_FREQ_BINS],
            ref_conv: vec![0.0_f32; CONV_SIZE],
            ref_tra: vec![0.0_f32; TRA_SIZE],
            ref_inter: vec![0.0_f32; INTER_SIZE],
        }
    }

    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.tra.fill(0.0);
        self.inter.fill(0.0);
    }

    /// Capture current state as the reference for decay.
    pub fn capture_reference(&mut self) {
        self.ref_conv.copy_from_slice(&self.conv);
        self.ref_tra.copy_from_slice(&self.tra);
        self.ref_inter.copy_from_slice(&self.inter);
    }

    /// Blend current state towards reference: state = state * alpha + ref * (1 - alpha)
    pub fn decay_towards_reference(&mut self, alpha: f32) {
        let one_minus = 1.0 - alpha;
        for (s, r) in self.conv.iter_mut().zip(self.ref_conv.iter()) {
            *s = *s * alpha + *r * one_minus;
        }
        for (s, r) in self.tra.iter_mut().zip(self.ref_tra.iter()) {
            *s = *s * alpha + *r * one_minus;
        }
        for (s, r) in self.inter.iter_mut().zip(self.ref_inter.iter()) {
            *s = *s * alpha + *r * one_minus;
        }
    }
}

impl Default for GtcrnState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Model Wrapper
// =============================================================================

pub struct GtcrnModel {
    model_type: ModelType,
    state: GtcrnState,
    session: Session,
    /// Exponential moving average of frame spectral energy.
    energy_avg: f32,
    /// Cooldown counter: while > 0, onset detection is inhibited.
    onset_cooldown: usize,
    /// Total frames processed; onset detection only activates after ONSET_MIN_FRAMES.
    frame_count: usize,
    /// True if previous frame's output was heavily suppressed (ratio < threshold).
    prev_over_suppressed: bool,
}

impl GtcrnModel {
    #[must_use]
    pub fn new(model_type: ModelType) -> Self {
        let model_bytes = match model_type {
            ModelType::Dns3 => EMBEDDED_MODEL_DNS3,
            ModelType::Vctk => EMBEDDED_MODEL_VCTK,
        };
        let session = create_session(model_bytes).expect("Failed to create ONNX session");
        let mut m = Self {
            model_type,
            state: GtcrnState::new(),
            session,
            energy_avg: 0.0,
            onset_cooldown: 0,
            frame_count: 0,
            prev_over_suppressed: false,
        };
        m.warm_up();
        m.state.capture_reference();
        m
    }

    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn set_model_type(&mut self, model_type: ModelType) {
        if self.model_type != model_type {
            let model_bytes = match model_type {
                ModelType::Dns3 => EMBEDDED_MODEL_DNS3,
                ModelType::Vctk => EMBEDDED_MODEL_VCTK,
            };
            match create_session(model_bytes) {
                Ok(new_session) => {
                    self.session = new_session;
                    self.model_type = model_type;
                    self.state.reset();
                    self.warm_up();
                    self.state.capture_reference();
                }
                Err(e) => {
                    eprintln!("GTCRN: Failed to switch model: {e}");
                }
            }
        }
    }

    /// Feed synthetic low-energy spectrum frames to bring recurrent state
    /// away from the over-suppressive all-zeros initialization.
    fn warm_up(&mut self) {
        let mut spectrum = [(0.0f32, 0.0f32); NUM_FREQ_BINS];
        // Deterministic low-energy "pink-ish" noise spectrum
        for (i, pair) in spectrum.iter_mut().enumerate() {
            let f = (i as f32 + 1.0) / NUM_FREQ_BINS as f32;
            let mag = 0.002 * (1.0 - f * 0.5);
            *pair = (mag, mag * 0.3);
        }
        for _ in 0..WARMUP_FRAMES {
            let _ = self.process_frame(&spectrum);
        }
    }

    pub fn process_frame(
        &mut self,
        spectrum: &[(f32, f32); NUM_FREQ_BINS],
    ) -> Result<[(f32, f32); NUM_FREQ_BINS], Box<dyn std::error::Error + Send + Sync>> {
        let input_energy: f32 = spectrum.iter().map(|&(re, im)| re * re + im * im).sum();

        self.frame_count += 1;
        if self.onset_cooldown > 0 {
            self.onset_cooldown -= 1;
        }
        self.energy_avg = self.energy_avg * (1.0 - ENERGY_SMOOTH) + input_energy * ENERGY_SMOOTH;

        // --- Pre-model onset detection ---
        // If the PREVIOUS frame was over-suppressed (ratio < 0.002 ≈ -27 dB)
        // and current input is a large energy jump, reset state BEFORE running
        // the model so this frame benefits from the reset immediately.
        if self.frame_count > ONSET_MIN_FRAMES
            && self.onset_cooldown == 0
            && self.prev_over_suppressed
            && input_energy > self.energy_avg * SUPPRESS_INPUT_FACTOR
        {
            self.state.decay_towards_reference(1.0 - ONSET_BLEND);
            self.onset_cooldown = ONSET_COOLDOWN_FRAMES;
        }

        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.state.input_buf[i * 2] = re;
            self.state.input_buf[i * 2 + 1] = im;
        }

        let input_tensor =
            TensorRef::from_array_view(([1usize, NUM_FREQ_BINS, 1, 2], &self.state.input_buf[..]))?;
        let conv_tensor =
            TensorRef::from_array_view(([2usize, 1, 16, 16, 33], &self.state.conv[..]))?;
        let tra_tensor = TensorRef::from_array_view(([2usize, 3, 1, 1, 16], &self.state.tra[..]))?;
        let inter_tensor =
            TensorRef::from_array_view(([2usize, 1, 33, 16], &self.state.inter[..]))?;

        let outputs = self.session.run(ort::inputs![
            input_tensor,
            conv_tensor,
            tra_tensor,
            inter_tensor,
        ])?;

        let (_, output_enh_data) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, output_conv_data) = outputs[1].try_extract_tensor::<f32>()?;
        let (_, output_tra_data) = outputs[2].try_extract_tensor::<f32>()?;
        let (_, output_inter_data) = outputs[3].try_extract_tensor::<f32>()?;

        self.state.conv.copy_from_slice(output_conv_data);
        self.state.tra.copy_from_slice(output_tra_data);
        self.state.inter.copy_from_slice(output_inter_data);

        for (i, pair) in self.state.output_buf.iter_mut().enumerate() {
            *pair = (output_enh_data[i * 2], output_enh_data[i * 2 + 1]);
        }

        // Track suppression for next frame's pre-model check
        let output_energy: f32 = self
            .state
            .output_buf
            .iter()
            .map(|&(re, im)| re * re + im * im)
            .sum();
        self.prev_over_suppressed =
            input_energy > 1e-10 && output_energy < input_energy * SUPPRESS_TRIGGER_RATIO;

        Ok(self.state.output_buf)
    }
}
