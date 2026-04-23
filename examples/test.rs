use std::ffi::CString;
use std::io::Write;

use rusty_llama::*;

fn main() {
    let _backend = Backend::acquire();
    println!("Backend initialized");

    let model = Model::load_from_file(
        "/Users/teo/Downloads/gemma-3-270m-it-Q8_0.gguf",
        ModelParams::new(),
    );

    unsafe {
        let mut desc = [0i8; 512];
        llama_sys::llama_model_desc(*model, desc.as_mut_ptr(), desc.len());
        let desc_str = std::ffi::CStr::from_ptr(desc.as_ptr()).to_string_lossy();
        println!("Model: {}", desc_str);
    }

    let vocab = unsafe { llama_sys::llama_model_get_vocab(*model) };
    let ctx = Context::new(&model, ContextParams::new());

    println!("Context initialized");

    // Tokenize prompt
    let prompt = "The future of AI is";
    let mut tokens = vec![0i32; 512];
    let n_tokens = unsafe {
        llama_sys::llama_tokenize(
            vocab,
            prompt.as_ptr() as *const i8,
            prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            true,
        )
    };

    println!("Tokens: {}", n_tokens);

    // Create batch
    let batch = unsafe { llama_sys::llama_batch_get_one(tokens.as_mut_ptr(), n_tokens) };

    // Decode prompt
    unsafe { llama_sys::llama_decode(*ctx, batch) };

    // Create sampler
    let sampler = unsafe { llama_sys::llama_sampler_init_greedy() };

    // Generate 10 tokens
    for _ in 0..10 {
        let token = unsafe { llama_sys::llama_sampler_sample(sampler, *ctx, -1) };
        let mut buf = [0u8; 64];
        let n = unsafe {
            llama_sys::llama_token_to_piece(
                vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                true,
            )
        };
        let piece = std::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        print!("{}", piece);
        std::io::stdout().flush().ok();

        let batch = unsafe { llama_sys::llama_batch_get_one(&token as *const i32 as *mut i32, 1) };
        unsafe { llama_sys::llama_decode(*ctx, batch) };
    }

    println!("Done!");
}
