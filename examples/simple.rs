use clap::Parser;
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
    let model_path = args.model;
    let prompt = args.prompt;
    let ngl = args.n_gpu_layers;
    let n_predict = args.n_predict;

    // backends get loaded and freed automatically

    // Initialize the model
    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = ngl;

    let model = Model::load_from_file(&model_path, model_params).expect("Failed to load model");
    println!("Model: {}", model.desc());

    let _vocab = model.vocab; // vocab is already in model struct and gets used automatically when needed

    // Tokenize the prompt
    let prompt_tokens = model.tokenize(&prompt, true, true);
    let n_prompt = prompt_tokens.len();

    // Initialize the context
    let mut ctx_params = ContextParams::new();
    // n_ctx is the context size
    ctx_params.n_ctx = (n_prompt + n_predict as usize - 1) as u32;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt as u32;
    // enable performance counters
    ctx_params.no_perf = false;

    let mut ctx = Context::new(&model, &ctx_params).expect("Failed to create context");
    println!("Context initialized");

    let mut seq = ctx.sequence().expect("failed to acquire sequence");
    seq.extend(&prompt_tokens);

    // Print the prompt token-by-token
    for token in &prompt_tokens {
        let piece = model.token_to_piece(*token).unwrap();
        print!("{}", piece);
    }
    std::io::stdout().flush().ok();

    // Main loop
    for _ in 0..n_predict {
        let (token, _) = seq.logits()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        if model.is_eog(token as i32) {
            break;
        }
        print!("{}", model.token_to_piece(token as i32).unwrap());
        seq.push(token as i32);
    }

    println!();

    let t_main_end = unsafe { llama_sys::llama_time_us() };

    drop(seq);
}
