use clap::Parser;
use llama_sys::*;
use rusty_llama::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the model file
    #[arg(short = 'm', long)]
    model: String,

    /// Number of tokens to predict
    #[arg(short = 'n', long, default_value_t = 32)]
    n_predict: i32,

    /// Number of parallel sequences
    #[arg(short = 'p', long = "parallel", default_value_t = 4)]
    n_parallel: i32,

    /// Number of GPU layers
    #[arg(long = "ngl", default_value_t = 99)]
    n_gpu_layers: i32,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.8)]
    temp: f32,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Top-p sampling parameter
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Seed for random number generator
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Prompt text
    #[arg(default_value = "Hello my name is")]
    prompt: String,

    /// Minimum keep for top-p sampling
    #[arg(long, default_value_t = 1)]
    min_keep: usize,

    /// Use backend sampling
    #[arg(long, default_value_t = false)]
    backend_sampling: bool,
}

fn main() {
    let args = Args::parse();

    let n_parallel = args.n_parallel;
    let n_predict = args.n_predict;
    let prompt = args.prompt;

    // backend init is handled for you

    // initialize the model
    let model = Model::load_from_file(&args.model, ModelParams::new()).unwrap();
    let _vocab = model.vocab;

    // tokenize the prompt
    let tokens_list = model.tokenize(&prompt, true, true);

    let n_kv_req =
        tokens_list.len() + (n_predict as usize - tokens_list.len()) * n_parallel as usize;

    // initialize the context
    let mut ctx_params = ContextParams::new();

    ctx_params.n_ctx = n_kv_req as u32;
    ctx_params.n_batch = std::cmp::max(n_predict, n_parallel) as u32;
    // set by common_context_params_to_llama()
    ctx_params.n_seq_max = n_parallel as u32;
    ctx_params.kv_unified = true;

    let mut sampler_params = SamplerChainParams::new();
    sampler_params.no_perf = false;

    let mut sampler_configs = vec![];

    for i in 0..n_parallel {
        let sampler = SamplerChain::new(&sampler_params)
            .add(Sampler::top_k(args.top_k))
            .add(Sampler::top_p(args.top_p, args.min_keep))
            .add(Sampler::temp(args.temp))
            // the original example doesn't vary the seed for some reason, but I did so here
            .add(Sampler::dist(args.seed + i as u32));

        sampler_configs.push(sampler);
    }

    // todo: backend sampling

    let mut ctx = Context::new(&model, &ctx_params).unwrap();

    let n_ctx = ctx.n_ctx();

    println!(
        "\nfunc: n_predict = {}, n_ctx = {}, n_batch = {}, n_parallel = {}, n_kv_req = {}",
        n_predict, n_ctx, &ctx_params.n_batch, n_parallel, n_kv_req
    );

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_ctx as usize {
        println!(
            "error: n_kv_req ({}) > n_ctx ({}), the required KV cache size is not big enough",
            n_kv_req, n_ctx
        );
        println!("        either reduce n_parallel or increase n_ctx");
        return;
    }

    // print the prompt token-by-token
    print!("\n");

    for token in &tokens_list {
        print!("{}", model.token_to_piece(*token).unwrap());
    }
    println!();

    // create a llama_batch
    // we use this object to submit token data for decoding
    let mut batch = unsafe {
        llama_sys::llama_batch_init(
            std::cmp::max(tokens_list.len(), n_parallel as usize) as i32,
            0,
            n_parallel,
        )
    };

    let mut seq_ids = vec![0; n_parallel as usize];
    for i in 0..n_parallel {
        seq_ids[i as usize] = i;
    }
    println!("seq_ids = {:?}", seq_ids);

    // evaluate the initial prompt
    println!("tokens_list.len() = {}", tokens_list.len());

    for (i, token) in tokens_list.iter().enumerate() {
        common::batch_add(&mut batch, *token, i as i32, seq_ids.clone(), false).unwrap();
    }
    assert!(batch.n_tokens == tokens_list.len() as i32);

    if model.has_encoder() {
        if ctx.encode(batch) != 0 {
            eprintln!("failed to eval");
            std::process::exit(1);
        }

        let decoder_start_token_id = model.decoder_start_token().unwrap();

        common::batch_clear(&mut batch);
        common::batch_add(
            &mut batch,
            decoder_start_token_id,
            0,
            seq_ids.clone(),
            false,
        )
        .unwrap();
    }

    // llama_decode will output logits only for the last token of the prompt
    unsafe {
        *batch.logits.add((batch.n_tokens - 1) as usize) = true as i8;
    }

    ctx.decode(batch).unwrap();

    if n_parallel > 1 {
        println!(
            "\n\n{}: generating {} sequences ...\n",
            std::file!(),
            n_parallel
        );
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    let mut streams: Vec<String> = vec![String::new(); n_parallel as usize];

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    let mut i_batch: Vec<i32> = vec![batch.n_tokens - 1; n_parallel as usize];

    let mut n_cur = batch.n_tokens;
    let mut n_decode = 0;

    let t_main_start = std::time::Instant::now();

    while n_cur <= n_predict {
        // prepare the next batch
        common::batch_clear(&mut batch);

        // sample the next token for each parallel sequence / stream
        for i in 0..n_parallel {
            if i_batch[i as usize] < 0 {
                // the stream has already finished
                continue;
            }

            let new_token_id = ctx.sample(&sampler_configs[i as usize], i_batch[i as usize]);

            // is it an end of generation? -> mark the stream as finished
            if model.is_eog(new_token_id) || n_cur == n_predict {
                i_batch[i as usize] = -1;
                println!();
                if n_parallel > 1 {
                    println!("stream {} finished at n_cur = {}", i, n_cur);
                }
                continue;
            }

            // if there is only one stream, we print immediately to stdout
            if n_parallel == 1 {
                println!("{}", model.token_to_piece(new_token_id).unwrap());
            }

            streams[i as usize] += &model.token_to_piece(new_token_id).unwrap();

            i_batch[i as usize] = batch.n_tokens;

            // push this new token for next evaluation
            common::batch_add(&mut batch, new_token_id, n_cur, vec![i], true).unwrap();

            n_decode += 1;
        }

        // all streams are finished
        if batch.n_tokens == 0 {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        ctx.decode(batch).unwrap();
    }

    if n_parallel > 1 {
        println!();
        for (i, stream) in streams.iter().enumerate() {
            println!("sequence {}:\n\n{}{}\n\n", i, prompt, stream);
        }
    }

    let t_main_end = std::time::Instant::now();

    println!(
        "\n\ndecoded {} tokens in {:.2} s, speed: {:.2} t/s\n",
        n_decode,
        (t_main_end - t_main_start).as_secs_f64(),
        n_decode as f64 / (t_main_end - t_main_start).as_secs_f64()
    );

    println!("{:?}", sampler_configs[0].perf());
    println!("{:?}", ctx.perf());

    unsafe {
        llama_batch_free(batch);
    }

    // the other drops should be handled for you already<3
}
