/// this is mostly a port of the simple_chat example from llama.cpp
/// ignoring the chat template parts
use clap::Parser;
use rusty_llama::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short = 'm', long)]
    model: String,

    #[arg(short = 'c', long, default_value_t = 2048)]
    context: u32,

    #[arg(long = "ngl", default_value_t = 99)]
    n_gpu_layers: i32,
}

fn main() {
    let args = Args::parse();
    let context = args.context;
    let n_gpu_layers = args.n_gpu_layers;
    let model_path = args.model;

    // backends get loaded and freed automatically

    // initialize the model
    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = n_gpu_layers;

    let model = Model::load_from_file(&model_path, model_params).expect("Failed to load model");

    // vocab is stored in the model
    let _vocab = model.vocab;

    // initialize the context
    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = context;
    ctx_params.n_batch = context;
    let mut ctx = Context::new(&model, &ctx_params).expect("Failed to create context");

    // initialize the sampler
    let smpl = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::min_p(0.05, 1))
        .add(Sampler::temp(0.8))
        .add(Sampler::dist(llama_sys::LLAMA_DEFAULT_SEED));

    // helper function to evaluate a prompt and generate a response
    let mut generate = |prompt: &str| {
        let mut response = String::new();

        let is_first = unsafe {
            llama_sys::llama_memory_seq_pos_max(llama_sys::llama_get_memory(ctx.as_ptr()), 0) == -1
        };

        // tokenize the prompt
        let mut tokens = model.tokenize(prompt, is_first, true);

        // prepare a batch for the prompt
        let mut batch =
            unsafe { llama_sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let mut new_token_id: i32;

        loop {
            // check if we have enough space in the context to evaluate this batch
            let n_ctx = unsafe { llama_sys::llama_n_ctx(ctx.as_ptr()) };
            let n_ctx_used = unsafe {
                llama_sys::llama_memory_seq_pos_max(llama_sys::llama_get_memory(ctx.as_ptr()), 0)
            } + 1;
            if n_ctx_used + batch.n_tokens > n_ctx as i32 {
                panic!("Prompt is too long");
            }

            ctx.decode(batch).expect("Failed to decode");

            // sample the next token
            new_token_id = ctx.sample(&smpl, -1);

            // is it an end of generation?
            if model.is_eog(new_token_id) || response.contains("\n") {
                break;
            }

            let piece = model
                .token_to_piece(new_token_id)
                .expect("failed to convert token to piece");
            print!("{}", piece);
            response.push_str(&piece);

            batch = unsafe { llama_sys::llama_batch_get_one(&mut new_token_id, 1) };
        }

        response
    };

    let mut messages: Vec<Message> = Vec::new();
    fn format(messages: &Vec<Message>) -> String {
        let mut s = messages
            .iter()
            .map(|m| match m {
                Message::User(s) => format!("user: {}", s),
                Message::Assistant(s) => format!("assistant: {}", s),
            })
            .collect::<Vec<String>>()
            .join("\n");
        s.push_str("\nassistant:");
        println!("{}", s);
        s
    }

    loop {
        // get user input
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        if input.is_empty() {
            break;
        }

        messages.push(Message::User(input.trim().to_string()));

        // generate a response
        println!();
        let response = generate(&format(&messages));
        println!();
        messages.push(Message::Assistant(response));
    }
}

enum Message {
    User(String),
    Assistant(String),
}
