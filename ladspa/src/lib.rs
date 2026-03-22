//! GTCRN LADSPA Plugin using ONNX Runtime with OpenVINO backend.
//!
//! This plugin provides real-time speech enhancement using the GTCRN neural network,
//! with ONNX Runtime as the inference backend. It can use OpenVINO as execution provider
//! for optimal CPU performance on Intel hardware.

pub mod biquad;
pub mod gate;
pub mod model;
pub mod plugin;
pub mod stft;

use ladspa::{ControlHint, DefaultValue, PluginDescriptor, Port, PortDescriptor, Properties};

/// Unique LADSPA plugin ID for GTCRN
pub const PLUGIN_ID: u64 = 0x4F52_5443; // "ORTC" in ASCII hex

/// Unique LADSPA plugin ID for the noise gate
pub const GATE_PLUGIN_ID: u64 = 9940;

/// Plugin version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Port index for audio input
pub const PORT_INPUT: usize = 0;

/// Port index for audio output
pub const PORT_OUTPUT: usize = 1;

/// Port index for enable control
pub const PORT_ENABLE: usize = 2;

/// Port index for strength control
pub const PORT_STRENGTH: usize = 3;

/// Port index for model selection control
pub const PORT_MODEL: usize = 4;

/// Port index for speech strength (dual-strength: filter intensity during detected speech)
pub const PORT_SPEECH_STRENGTH: usize = 5;

/// Port index for lookahead in milliseconds (output delay enabling pre-speech detection)
pub const PORT_LOOKAHEAD_MS: usize = 6;

/// Port index for model blending control (0=off, 1=dual-model VAD-switched)
pub const PORT_MODEL_BLEND: usize = 7;

/// Port index for voice recovery level (>8kHz reconstruction)
/// 0.0 = cut all HF, 0.7 = natural default, 1.0 = full original HF
pub const PORT_VOICE_RECOVERY: usize = 8;

/// Returns the LADSPA plugin descriptor.
#[no_mangle]
#[allow(unsafe_code)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    match index {
        0 => Some(gtcrn_descriptor()),
        1 => Some(gate_descriptor()),
        _ => None,
    }
}

/// GTCRN speech enhancement plugin descriptor.
fn gtcrn_descriptor() -> PluginDescriptor {
    PluginDescriptor {
        unique_id: PLUGIN_ID,
        label: "gtcrn_mono",
        properties: Properties::PROP_REALTIME,
        name: "GTCRN Speech Enhancement (ORT)",
        maker: "GTCRN Model (c) 2024 Rong Xiaobin | Ladspa plugin (c) 2026 Bruno Gonçalves",
        copyright: "MIT License",
        ports: vec![
            Port {
                name: "Input",
                desc: PortDescriptor::AudioInput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Output",
                desc: PortDescriptor::AudioOutput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Enable",
                desc: PortDescriptor::ControlInput,
                hint: None, // NOT HINT_TOGGLED: PipeWire 1.6.x sets toggled defaults to 0
                default: Some(DefaultValue::Maximum), // 1.0
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "Strength",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Value1),
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "Model",
                desc: PortDescriptor::ControlInput,
                hint: Some(ControlHint::HINT_INTEGER),
                default: Some(DefaultValue::Value0), // 0 = DNS3 (default, strongest NR)
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "SpeechStrength",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Maximum), // 1.0 — Voice Preservation 100%
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "LookaheadMs",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~50 ms
                lower_bound: Some(0.0),
                upper_bound: Some(200.0),
            },
            Port {
                name: "ModelBlend",
                desc: PortDescriptor::ControlInput,
                hint: None, // NOT HINT_TOGGLED: PipeWire 1.6.x sets toggled defaults to 0
                default: Some(DefaultValue::Minimum), // 0.0 — OFF: dual-model blend introduces artifacts (v4 test)
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "VoiceRecovery",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::High), // 0.75 — natural HF level
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
        ],
        new: plugin::GtcrnPlugin::new,
    }
}

/// Noise gate with sidechain bandpass key filter descriptor.
pub fn gate_descriptor() -> PluginDescriptor {
    PluginDescriptor {
        unique_id: GATE_PLUGIN_ID,
        label: "biglinux_gate",
        properties: Properties::PROP_REALTIME,
        name: "BigLinux Noise Gate (Bandpass Key Filter)",
        maker: "(c) 2025 Bruno Gonçalves",
        copyright: "MIT License",
        ports: vec![
            Port {
                name: "Input",
                desc: PortDescriptor::AudioInput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Output",
                desc: PortDescriptor::AudioOutput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Threshold (dB)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~ -60 dB
                lower_bound: Some(-80.0),
                upper_bound: Some(0.0),
            },
            Port {
                name: "Attack (ms)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~10 ms
                lower_bound: Some(0.1),
                upper_bound: Some(500.0),
            },
            Port {
                name: "Hold (ms)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Middle), // ~1000 ms
                lower_bound: Some(0.0),
                upper_bound: Some(2000.0),
            },
            Port {
                name: "Release (ms)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~200 ms
                lower_bound: Some(1.0),
                upper_bound: Some(2000.0),
            },
            Port {
                name: "Range (dB)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Minimum), // -90 dB
                lower_bound: Some(-90.0),
                upper_bound: Some(0.0),
            },
            Port {
                name: "LF Key Filter (Hz)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~200 Hz
                lower_bound: Some(20.0),
                upper_bound: Some(20000.0),
            },
            Port {
                name: "HF Key Filter (Hz)",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::Low), // ~4000 Hz
                lower_bound: Some(20.0),
                upper_bound: Some(20000.0),
            },
        ],
        new: gate::GatePlugin::new,
    }
}

/// Re-export for LADSPA host discovery
pub use ladspa::ladspa_descriptor;
