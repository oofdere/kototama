use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let llama_cpp = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("llama.cpp");

    let include_path = llama_cpp.join("include");
    let ggml_path = llama_cpp.join("ggml/include");

    // Build llama.cpp if not already built
    let build_dir = llama_cpp.join("build");
    let llama_lib = build_dir.join("src/libllama.a");
    if !llama_lib.exists() {
        std::fs::create_dir_all(&build_dir).unwrap();
        
        // Configure with CMake
        let status = Command::new("cmake")
            .current_dir(&build_dir)
            .args([
                "..",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DLLAMA_METAL=ON",
            ])
            .status()
            .expect("Failed to run cmake. Make sure cmake is installed.");
        
        if !status.success() {
            panic!("cmake configuration failed");
        }
        
        // Build
        let status = Command::new("cmake")
            .current_dir(&build_dir)
            .args(["--build", ".", "--", "-j"])
            .status()
            .expect("Failed to build llama.cpp");
        
        if !status.success() {
            panic!("llama.cpp build failed");
        }
    }

    // Link the static libraries
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}/src", build_dir.display());
    println!("cargo:rustc-link-search=native={}/ggml/src", build_dir.display());
    println!("cargo:rustc-link-search=native={}/ggml/src/ggml-cpu", build_dir.display());
    println!("cargo:rustc-link-search=native={}/ggml/src/ggml-metal", build_dir.display());
    println!("cargo:rustc-link-search=native={}/ggml/src/ggml-blas", build_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-metal");
    println!("cargo:rustc-link-lib=static=ggml-blas");
    
    // Platform-specific libraries
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=c++");
    }
    
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    }

    // Tell cargo to invalidate the built crate whenever the header changes
    println!("cargo:rerun-if-changed={}/llama.h", include_path.display());

    // Generate bindings with all necessary includes
    let bindings = bindgen::Builder::default()
        .header(format!("{}/llama.h", include_path.display()))
        .clang_arg(format!("-I{}", include_path.display()))
        .clang_arg(format!("-I{}", ggml_path.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .derive_debug(true)
        .clang_arg("-fparse-all-comments")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");
}
