/// Port of the simple_chat example from llama.cpp, using the actor-based API.
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

    let mut model_params = ModelParams::new();
    model_params.n_gpu_layers = args.n_gpu_layers;

    let model = Model::load_from_file(&args.model, model_params).expect("Failed to load model");

    let mut ctx_params = ContextParams::new();
    ctx_params.n_ctx = args.context;
    ctx_params.n_batch = args.context;

    let ctx = Context::new(&model, &ctx_params).expect("Failed to create context");
    let mut seq = ctx.sequence().expect("failed to acquire sequence");

    let smpl = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::min_p(0.05, 1))
        .add(Sampler::temp(0.8))
        .add(Sampler::dist(llama_sys::LLAMA_DEFAULT_SEED));

    let mut messages: Vec<Message> = Vec::new();

    fn format_messages(messages: &[Message]) -> String {
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
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        if input.is_empty() {
            break;
        }

        messages.push(Message::User(input.trim().to_string()));

        let prompt = format_messages(&messages);
        let is_first = seq.is_empty();
        let tokens = model.tokenize(&prompt, is_first, true);

        seq.extend(&tokens);

        let mut response = String::new();
        println!();
        loop {
            let token = seq.sample(&smpl);

            if model.is_eog(token) || response.contains('\n') {
                break;
            }

            let piece = model
                .token_to_piece(token)
                .expect("failed to convert token to piece");
            print!("{}", piece);
            response.push_str(&piece);

            seq.push(token);
        }
        println!();
        messages.push(Message::Assistant(response));
    }
}

enum Message {
    User(String),
    Assistant(String),
}
