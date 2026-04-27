//! External control file for live LADSPA parameter updates.
//!
//! The host UI writes a small binary file under `$XDG_RUNTIME_DIR` so the
//! plugin can pick up parameter changes without re-instantiating the LADSPA
//! chain. The file lives on tmpfs, so reads are page-cache lookups
//! (~1 μs) — but we still throttle to once every [`EXTERNAL_POLL_INTERVAL`]
//! `run()` calls to avoid taking the syscall hit on every audio block.
//!
//! Format (six little-endian f32 values, 24 bytes total):
//! ```text
//! [strength, model, speech_strength, lookahead_ms, model_blend, voice_recovery]
//! ```
//! Older 8-byte and 16-byte payloads are accepted for backward compatibility
//! — missing fields keep their default of `-1.0` (meaning "use LADSPA port").

/// Throttle factor: re-read the control file every Nth `run()` call.
///
/// At 48 kHz with a 1024-sample block this is ~46 Hz of polling — fast
/// enough that GUI sliders feel live, slow enough that the syscall cost is
/// negligible on the audio thread.
pub const EXTERNAL_POLL_INTERVAL: u32 = 10;

/// Filename inside `$XDG_RUNTIME_DIR` (tmpfs) where the GUI writes the
/// live parameter values for this plugin instance to read.
pub const CONTROL_FILENAME: &str = "gtcrn-ladspa-controls";

/// Snapshot of the most-recently-read live parameter values.
///
/// `-1.0` is the sentinel meaning "field not present in the file → fall
/// back to the LADSPA port value", except for `model_blend` whose sentinel
/// is `1.0` (legacy behaviour: assume blend on when the field is missing).
pub struct ExternalControls {
    path: std::path::PathBuf,
    pub strength: f32,
    pub model_type: f32,
    pub speech_strength: f32,
    pub lookahead_ms: f32,
    pub model_blend: f32,
    pub voice_recovery: f32,
    pub available: bool,
    counter: u32,
}

impl ExternalControls {
    /// Build a controller bound to `$XDG_RUNTIME_DIR/gtcrn-ladspa-controls`,
    /// falling back to `/tmp` when the env var is missing (sandbox / CI only —
    /// `/tmp` is world-writable, so a desktop session must always provide
    /// `XDG_RUNTIME_DIR` for the per-user 0700 directory).
    #[must_use]
    pub fn new() -> Self {
        let path = std::env::var("XDG_RUNTIME_DIR")
            .map_or_else(
                |_| std::path::PathBuf::from("/tmp"),
                std::path::PathBuf::from,
            )
            .join(CONTROL_FILENAME);

        let mut ctrl = Self::with_initial_path(path);
        ctrl.poll();
        ctrl
    }

    fn with_initial_path(path: std::path::PathBuf) -> Self {
        Self {
            path,
            strength: -1.0,
            model_type: -1.0,
            speech_strength: -1.0,
            lookahead_ms: -1.0,
            model_blend: 1.0,
            voice_recovery: -1.0,
            available: false,
            counter: EXTERNAL_POLL_INTERVAL, // trigger immediate read
        }
    }

    /// Test-only constructor with a custom file path. The constructor
    /// performs no immediate poll; tests drive `poll()` explicitly.
    #[cfg(test)]
    #[must_use]
    pub fn with_path(path: std::path::PathBuf) -> Self {
        Self::with_initial_path(path)
    }

    /// Re-read the control file when enough `run()` calls have elapsed.
    ///
    /// The throttle exists so the audio thread doesn't hit the kernel
    /// every block — the file lives on tmpfs but `read()` still walks
    /// the dentry cache and acquires a per-inode lock.
    pub fn poll(&mut self) {
        self.counter += 1;
        if self.counter < EXTERNAL_POLL_INTERVAL {
            return;
        }
        self.counter = 0;
        let Ok(data) = std::fs::read(&self.path) else {
            return;
        };
        // Format v7: 24 bytes (6 f32 LE values)
        // [strength, model, speech_strength, lookahead_ms, model_blend, voice_recovery]
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

impl Default for ExternalControls {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_control_file(path: &std::path::Path, data: &[u8]) {
        let mut file = std::fs::File::create(path).expect("create control file");
        file.write_all(data).expect("write control file");
    }

    fn tempdir() -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gtcrn-extctrl-test-{id}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Drive `poll()` enough times to trigger an actual read. With the
    /// throttle bumped to 10, the original test pattern of "construct
    /// then poll once" stops working — we now need to advance the
    /// counter past `EXTERNAL_POLL_INTERVAL`.
    fn poll_until_read(ctrl: &mut ExternalControls) {
        for _ in 0..=EXTERNAL_POLL_INTERVAL {
            ctrl.poll();
        }
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
        poll_until_read(&mut ctrl);
        assert!(ctrl.available);
        assert!((ctrl.strength - 0.75).abs() < 1e-6);
        assert!((ctrl.model_type - 1.0).abs() < 1e-6);
        assert!((ctrl.speech_strength - -1.0).abs() < 1e-6);
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
        poll_until_read(&mut ctrl);
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
        poll_until_read(&mut ctrl);
        assert!(ctrl.available);
        assert!((ctrl.strength - 0.9).abs() < 1e-6);
        assert!((ctrl.model_type - 0.0).abs() < 1e-6);
        assert!((ctrl.speech_strength - 1.0).abs() < 1e-6);
        assert!((ctrl.lookahead_ms - 50.0).abs() < 1e-6);
        assert!((ctrl.model_blend - 0.5).abs() < 1e-6);
        assert!((ctrl.voice_recovery - 0.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poll_missing_file() {
        let path = std::path::PathBuf::from("/tmp/gtcrn-test-nonexistent-12345");
        let mut ctrl = ExternalControls::with_path(path);
        poll_until_read(&mut ctrl);
        assert!(!ctrl.available);
    }

    #[test]
    fn poll_short_file_ignored() {
        let dir = tempdir();
        let path = dir.join("controls");
        write_control_file(&path, &[0_u8; 4]); // <8 bytes — not parsed

        let mut ctrl = ExternalControls::with_path(path);
        poll_until_read(&mut ctrl);
        assert!(!ctrl.available);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn throttle_holds_off_reads_between_polls() {
        let dir = tempdir();
        let path = dir.join("controls");
        let mut ctrl = ExternalControls::with_path(path.clone());

        // No file yet — even after polling enough to trigger reads, the
        // tracker stays unavailable because every read fails.
        for _ in 0..EXTERNAL_POLL_INTERVAL * 2 {
            ctrl.poll();
        }
        assert!(!ctrl.available);

        // Drop a valid file in place. A single poll after this may or
        // may not trigger a read depending on where the counter sits;
        // drive at least one full poll period to guarantee we cross
        // the threshold.
        let strength: f32 = 0.5;
        let model: f32 = 0.0;
        let mut data = Vec::new();
        data.extend_from_slice(&strength.to_le_bytes());
        data.extend_from_slice(&model.to_le_bytes());
        write_control_file(&path, &data);

        for _ in 0..=EXTERNAL_POLL_INTERVAL {
            ctrl.poll();
        }
        assert!(ctrl.available);
        std::fs::remove_dir_all(&dir).ok();
    }
}
