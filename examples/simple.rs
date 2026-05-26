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

    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = ngl;

    let model = Model::load_from_file(&model_path, model_params).expect("Failed to load model");
    println!("Model: {}", model.desc());

    if model.has_encoder() {
        panic!("Model has encoder, which is not supported in this example");
    }

    if n_predict <= 0 {
        eprintln!("n_predict must be positive, got {n_predict}");
        std::process::exit(1);
    }

    let prompt_tokens = model.tokenize(&prompt, true, true);
    let n_prompt = prompt_tokens.len();

    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = (n_prompt + n_predict as usize - 1) as u32;
    ctx_params.n_batch = 1;
    ctx_params.no_perf = false;

    let ctx = Context::new(&model, &ctx_params).expect("Failed to create context");

    let mut seq = ctx.sequence().expect("failed to acquire sequence");
    seq.extend(&prompt_tokens);

    for token in &prompt_tokens {
        let piece = model.token_to_piece(*token).unwrap();
        print!("{}", piece);
    }
    std::io::stdout().flush().ok();

    // Main loop — greedy argmax from cached logits
    for _ in 0..n_predict {
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
        seq.push(token as i32);
    }

    println!();
}
