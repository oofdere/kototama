use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    let mut config = Config::new("llama.cpp");

    // Core settings
    config
        .define("BUILD_SHARED_LIBS", "OFF") // static build
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_PERF", "OFF");
    //.define("GGML_LTO", "ON");

    #[cfg(feature = "native")]
    {
        config.define("GGML_NATIVE", "ON");
    }

    #[cfg(feature = "vulkan")]
    {
        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            config.cflag(&format!("-I{}/include", vulkan_sdk));
            config.cxxflag(&format!("-I{}/include", vulkan_sdk));
        } else {
            panic!("VULKAN_SDK not set, see llama.cpp docs for config info");
        }
        config.define("GGML_VULKAN", "ON");
    }

    #[cfg(feature = "cuda")]
    {
        config.define("LLAMA_CUDA", "ON");
    }

    #[cfg(all(target_os = "macos"))]
    {
        config.define("GGML_METAL", "ON");
        config.define("GGML_METAL_NDEBUG", "ON");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    let dst = config.build();

    // Link the libraries (order matters - dependencies after dependents)
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=llama-common");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-metal");
        println!("cargo:rustc-link-lib=static=ggml-blas");
        println!("cargo:rustc-link-lib=c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml-cpu");
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml");
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml-base");

        #[cfg(feature = "vulkan")]
        {
            println!("cargo:rustc-link-lib=static:+whole-archive=ggml-vulkan");
            println!("cargo:rustc-link-lib=vulkan");
        }

        #[cfg(feature = "cuda")]
        {
            println!("cargo:rustc-link-lib=static:+whole-archive=ggml-cuda");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        }

        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rebuild triggers
    println!("cargo:rerun-if-changed=llama.cpp/");

    // Generate bindings with all necessary includes
    let bindings = bindgen::Builder::default()
        .header("llama.cpp/include/llama.h")
        .clang_arg("-Illama.cpp/include")
        .clang_arg("-Illama.cpp/ggml/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .derive_debug(true)
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .rustified_non_exhaustive_enum("ggml_type")
        .rustified_non_exhaustive_enum("ggml_backend_buffer_usage")
        .rustified_non_exhaustive_enum("ggml_backend_dev_type")
        .rustified_non_exhaustive_enum("ggml_backend_meta_split_axis")
        .rustified_non_exhaustive_enum("ggml_ftype")
        .rustified_non_exhaustive_enum("ggml_glu_op")
        .rustified_non_exhaustive_enum("ggml_log_level")
        .rustified_non_exhaustive_enum("ggml_numa_strategy")
        .rustified_non_exhaustive_enum("ggml_object_type")
        .rustified_non_exhaustive_enum("ggml_op")
        .rustified_non_exhaustive_enum("ggml_opt_build_type")
        .rustified_non_exhaustive_enum("ggml_opt_loss_type")
        .rustified_non_exhaustive_enum("ggml_opt_optimizer_type")
        .rustified_non_exhaustive_enum("ggml_prec")
        .rustified_non_exhaustive_enum("ggml_scale_flag")
        .rustified_non_exhaustive_enum("ggml_scale_mode")
        .rustified_non_exhaustive_enum("ggml_sched_priority")
        .rustified_non_exhaustive_enum("ggml_sort_order")
        .rustified_non_exhaustive_enum("ggml_status")
        .rustified_non_exhaustive_enum("ggml_tensor_flag")
        .rustified_non_exhaustive_enum("ggml_tri_type")
        .rustified_non_exhaustive_enum("ggml_unary_op")
        .rustified_non_exhaustive_enum("gguf_type")
        .rustified_non_exhaustive_enum("llama_attention_type")
        .rustified_non_exhaustive_enum("llama_flash_attn_type")
        .rustified_non_exhaustive_enum("llama_ftype")
        .rustified_non_exhaustive_enum("llama_model_kv_override_type")
        .rustified_non_exhaustive_enum("llama_model_meta_key")
        .rustified_non_exhaustive_enum("llama_pooling_type")
        .rustified_non_exhaustive_enum("llama_rope_scaling_type")
        .rustified_non_exhaustive_enum("llama_rope_type")
        .rustified_non_exhaustive_enum("llama_split_mode")
        .rustified_non_exhaustive_enum("llama_token_attr")
        .rustified_non_exhaustive_enum("llama_token_type")
        .rustified_non_exhaustive_enum("llama_vocab_type")
        .rustified_non_exhaustive_enum("ggml_op_pool")
        .rustified_non_exhaustive_enum("GGML_ROPE_TYPE")
        .layout_tests(false)
        .clang_arg("-fparse-all-comments")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");

    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");
}
