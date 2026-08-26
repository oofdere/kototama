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

    // initialize the context
    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = context;
    ctx_params.n_batch = context;

    let ctx = Context::new(&model, &ctx_params).expect("Failed to create context");
    let mut seq = ctx.sequence().expect("failed to acquire sequence");

    // initialize the samplers (applied manually: min_p -> temp -> dist)
    let mut minp = MinP::new(0.05, 1);
    let mut temp = Temperature::new(0.8);
    let mut dist = Dist::new(llama_sys::LLAMA_DEFAULT_SEED as u64);

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
        let prompt = format(&messages);
        let is_first = seq.is_empty();

        // tokenize the prompt
        let tokens = model.tokenize(&prompt, is_first, true);

        seq.extend(&tokens).unwrap();

        let mut response = String::new();
        println!();
        loop {
            // sample the next token (min_p -> temp -> dist)
            let logits = seq.logits().expect("no logits");
            let l = minp.apply(logits);
            let l = temp.apply(&l);
            let token = dist.sample(&l);

            // is it an end of generation?
            if model.is_eog(token) {
                break;
            }

            let piece = model
                .token_to_piece(token)
                .expect("failed to convert token to piece");
            print!("{}", piece);
            response.push_str(&piece);

            seq.push(token).unwrap();

            if piece.contains('\n') {
                break;
            }
        }
        println!();
        messages.push(Message::Assistant(response));
    }
}

enum Message {
    User(String),
    Assistant(String),
}
