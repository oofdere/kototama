use rusty_llama::*;
use std::io::Write;
use clap::{Parser};

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

    let model = Model::load_from_file(&model_path, model_params); // ! THIS MIGHT RETURN NULL AND EXPLODE
    println!("Model: {}", model.desc());

    let _vocab = model.vocab; // vocab is already in model struct and gets used automatically when needed

    // Tokenize the prompt
    let prompt_tokens = model.tokenize(&prompt);
    let n_prompt = prompt_tokens.len();

    // Initialize the context
    let mut ctx_params = ContextParams::new();
    // n_ctx is the context size
    ctx_params.n_ctx = (n_prompt + n_predict as usize - 1) as u32;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt as u32;
    // enable performance counters
    ctx_params.no_perf = false;

    let mut ctx = Context::new(&model, ctx_params);
    println!("Context initialized");

    // Initialize the sampler
    let mut sparams = SamplerChainParams::new();
    sparams.no_perf = false;
    let sampler = SamplerChain::new().add(Sampler::greedy());

    // Print the prompt token-by-token
    for token in &prompt_tokens {
        let piece = model.token_to_piece(*token);
        print!("{}", piece);
    }
    std::io::stdout().flush().ok();

    // Prepare a batch for the prompt
    let mut batch = unsafe {
        llama_sys::llama_batch_get_one(prompt_tokens.as_ptr() as *mut i32, n_prompt as i32)
    }; // this is depracated upstream

    if model.has_encoder() {
        if ctx.encode(batch) != 0 {
            eprintln!("failed to eval");
            std::process::exit(1);
        }

        let decoder_start_token_id = model.decoder_start_token();
        let decoder_start_token_id = if decoder_start_token_id == -1 {
            model.bos_token()
        } else {
            decoder_start_token_id
        };

        batch = unsafe {
            llama_sys::llama_batch_get_one((&decoder_start_token_id) as *const i32 as *mut i32, 1)
        };
    }

    // Main loop
    let t_main_start = unsafe { llama_sys::llama_time_us() };
    let mut n_decode = 0;
    let mut new_token_id: i32;

    let mut n_pos = 0i32;
    let max_tokens = (n_prompt + n_predict as usize) as i32;
    while n_pos + batch.n_tokens < max_tokens {
        // Evaluate the current batch with the transformer model
        if ctx.decode(batch) != 0 {
            eprintln!("failed to eval, return code 1");
            std::process::exit(1);
        }

        n_pos += batch.n_tokens;

        // Sample the next token
        new_token_id = ctx.sample(&sampler, -1);

        // Is it an end of generation?
        if model.is_eog(new_token_id) {
            break;
        }

        let piece = model.token_to_piece(new_token_id);
        print!("{}", piece);
        std::io::stdout().flush().ok();

        // Prepare the next batch with the sampled token
        batch =
            unsafe { llama_sys::llama_batch_get_one((&new_token_id) as *const i32 as *mut i32, 1) };

        n_decode += 1;
    }

    println!();

    let t_main_end = unsafe { llama_sys::llama_time_us() };

    println!(
        "decoded {} tokens in {:.2} s, speed: {:.2} t/s",
        n_decode,
        (t_main_end - t_main_start) as f64 / 1_000_000.0,
        n_decode as f64 / ((t_main_end - t_main_start) as f64 / 1_000_000.0)
    );

    println!("{:?}", sampler.perf());
    println!("{:?}", ctx.perf());
}
