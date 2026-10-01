//! Async version of the `simple` example: generate tokens from a prompt using
//! the `*_async` methods, driven by the built-in `block_on` executor.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example simple_async -- -m ./test-models/TinyStories-656K.Q2_K.gguf
//! ```

use clap::Parser;
use rusty_llama::asynchronous::block_on;
use rusty_llama::*;
use std::io::Write;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the model file
    #[arg(short = 'm', long)]
    model: String,

    /// Number of tokens to predict
    #[arg(short = 'n', long, default_value_t = 32)]
    n_predict: i32,

    /// Number of GPU layers
    #[arg(long = "ngl", default_value_t = 99)]
    n_gpu_layers: i32,

    /// Prompt text
    #[arg(default_value = "Hello my name is")]
    prompt: String,
}

fn main() {
    let args = Args::parse();
    if args.n_predict <= 0 {
        eprintln!("n_predict must be positive, got {}", args.n_predict);
        std::process::exit(1);
    }

    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = args.n_gpu_layers;

    // Model and context creation are blocking FFI calls, so they go through
    // the async facade just like decoding does.
    let model =
        block_on(Model::load_from_file_async(&args.model, model_params)).expect("failed to load model");
    println!("Model: {}", model.desc());

    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = (args.n_predict + 1) as u32;
    ctx_params.n_batch = 1;
    ctx_params.no_perf = false;

    let ctx = block_on(Context::new_async(&model, &ctx_params)).expect("failed to create context");
    let mut seq = ctx.sequence().expect("failed to acquire sequence");

    let prompt_tokens = model.tokenize(&args.prompt, true, true);
    block_on(seq.extend_async(&prompt_tokens)).expect("failed to decode prompt");

    // Echo the prompt.
    for token in &prompt_tokens {
        print!("{}", model.token_to_piece(*token).unwrap());
    }
    std::io::stdout().flush().ok();

    // Generation loop: sampling stays synchronous (it reads the cached
    // logits), only decoding is offloaded to the thread pool.
    for _ in 0..args.n_predict {
        let (token, _) = seq
            .logits()
            .expect("no logits")
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        if model.is_eog(token as i32) {
            break;
        }
        print!("{}", model.token_to_piece(token as i32).unwrap());
        std::io::stdout().flush().ok();
        block_on(seq.push_async(token as i32)).expect("decode failed");
    }

    println!();
}
