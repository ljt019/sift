use std::env;
use std::path::PathBuf;

use cudaforge::{KernelBuilder, Result};

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR should be set"));
    let mut builder = KernelBuilder::new()
        .source_dir("src")
        // CUDA 13 supports Turing and newer. Embedding PTX for its baseline
        // virtual architecture lets the installed driver JIT the kernels for
        // whichever supported GPU runs Sift.
        .compute_cap(75)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Nix exposes CUDA components through a merged toolkit path rather than
    // /usr/local/cuda. Supplying the root also makes the same build work with a
    // conventional CUDA installation when CUDA_ROOT is set explicitly.
    let include_arg;
    if let Ok(root) = env::var("CUDA_ROOT") {
        include_arg = format!("-I{root}/include");
        builder = builder.cuda_root(&root).arg(&include_arg);
    }

    builder.build_ptx()?.write(output.join("ptx.rs"))?;
    Ok(())
}
