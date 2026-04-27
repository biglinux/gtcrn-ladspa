// Build script:
//   1. Convert embedded ONNX models to .ort (every build).
//   2. For the `static` feature only, link the bundled minimal ONNX
//      Runtime + Abseil archives produced by build-minimal-docker.sh.
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    convert_models(&manifest_dir);

    if !cfg!(feature = "static") {
        return;
    }

    let lib_dir = PathBuf::from(&manifest_dir)
        .join("onnxruntime-minimal")
        .join("lib");

    assert!(
        lib_dir.exists(),
        "Static ONNX Runtime libraries not found at {}. Run ./build-minimal-docker.sh first.",
        lib_dir.display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link the unified libonnxruntime.a composed by build.sh.
    // `+whole-archive` ensures symbols like OrtGetApiBase are included even
    // when not referenced by Rust (the ort crate initialises them lazily).
    println!("cargo:rustc-link-lib=static:+whole-archive=onnxruntime");

    // Auto-discover and link Abseil libraries (libabsl_*.a). Sort for a
    // deterministic build order; cyclic deps may need link groups instead.
    let mut absl_libs = Vec::new();
    if let Ok(entries) = fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap();
            let is_static_lib = std::path::Path::new(&name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("a"));
            if name.starts_with("libabsl_") && is_static_lib {
                let lib_name = &name[3..name.len() - 2]; // strip "lib" prefix + ".a" suffix
                absl_libs.push(lib_name.to_string());
            }
        }
    }
    absl_libs.sort();

    for lib in absl_libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // Link other static deps that build-minimal-docker.sh emits.
    let deps = [
        "protobuf",
        "protobuf-lite",
        "nsync_cpp",
        "cpuinfo",
        "flatbuffers",
    ];
    for dep in deps {
        let filename = format!("lib{dep}.a");
        if lib_dir.join(&filename).exists() {
            println!("cargo:rustc-link-lib=static={dep}");
        }
    }

    // System libraries pulled in by the static archives.
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=stdc++");
}

fn convert_models(manifest_dir: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let models = [
        ("gtcrn_simple.onnx", "gtcrn_simple.ort"),
        ("gtcrn_vctk.onnx", "gtcrn_vctk.ort"),
    ];

    let stream_dir = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("stream")
        .join("onnx_models");
    // GTCRN_PYTHON env var lets packagers (PKGBUILD) point at a system
    // python without patching this file. Default keeps the local .venv flow.
    println!("cargo:rerun-if-env-changed=GTCRN_PYTHON");
    let python_path = env::var_os("GTCRN_PYTHON").map_or_else(
        || PathBuf::from(manifest_dir).join(".venv/bin/python"),
        PathBuf::from,
    );

    let python_in_path = python_path.components().count() == 1;
    if !python_in_path && !python_path.exists() {
        println!(
            "cargo:warning=Python interpreter not found at {}, skipping model conversion.",
            python_path.display()
        );
        return;
    }

    for (src_name, dst_name) in models {
        let src_path = stream_dir.join(src_name);
        let dst_path = out_dir.join(dst_name);

        println!("cargo:rerun-if-changed={}", src_path.display());

        if !src_path.exists() {
            println!(
                "cargo:warning=Source model not found at {}, skipping conversion.",
                src_path.display()
            );
            continue;
        }

        let should_convert = if dst_path.exists() {
            let src_meta = fs::metadata(&src_path).unwrap();
            let dst_meta = fs::metadata(&dst_path).unwrap();
            src_meta.modified().unwrap() > dst_meta.modified().unwrap()
        } else {
            true
        };

        if should_convert {
            println!("Converting {src_name} to ORT format...");
            let status = std::process::Command::new(&python_path)
                .args([
                    "-m",
                    "onnxruntime.tools.convert_onnx_models_to_ort",
                    src_path.to_str().unwrap(),
                    "--output_dir",
                    out_dir.to_str().unwrap(),
                    "--optimization_style",
                    "Fixed",
                ])
                .status()
                .expect("Failed to run conversion command");

            assert!(status.success(), "Model conversion failed for {src_name}");
        }
    }
}
