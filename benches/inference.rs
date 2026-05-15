use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use rusty_llama::{Context, ContextParams, Model, ModelParams};

fn bench_model_path() -> String {
    std::env::var("RUSTY_LLAMA_BENCH_MODEL")
        .or_else(|_| std::env::var("RUSTY_LLAMA_TEST_MODEL"))
        .unwrap_or_else(|_| "./test-models/smollm-135m.gguf".to_string())
}

fn load_model() -> Model {
    let path = bench_model_path();
    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    Model::load_from_file(&path, params)
        .unwrap_or_else(|_| panic!("Failed to load model from '{}'. Set RUSTY_LLAMA_BENCH_MODEL.", path))
}

fn make_ctx_params() -> ContextParams {
    let mut p = ContextParams::new();
    p.n_ctx = 512;
    p.n_batch = 512;
    p.n_seq_max = 1;
    p.no_perf = true;
    p
}

fn bench_tokenize(c: &mut Criterion) {
    let model = load_model();
    let prompt = "The quick brown fox jumps over the lazy dog.";
    c.bench_function("tokenize", |b| {
        b.iter(|| model.tokenize(black_box(prompt), false, false))
    });
}

fn bench_token_to_piece(c: &mut Criterion) {
    let model = load_model();
    let bos = model.bos_token().unwrap_or(1);
    c.bench_function("token_to_piece", |b| {
        b.iter(|| model.token_to_piece(black_box(bos)))
    });
}

fn bench_decode_single_token(c: &mut Criterion) {
    let model = load_model();
    let ctx_params = make_ctx_params();

    c.bench_function("decode_single_token", |b| {
        b.iter_batched(
            || {
                // Leak ctx so that Sequence (which borrows it) can be returned
                // from this closure. The Box is recovered and dropped in the
                // measurement closure via `Box::from_raw`.
                let ctx = Box::new(Context::new(&model, &ctx_params).unwrap());
                let ctx_ref: &'static Context<'static> =
                    unsafe { &*(Box::into_raw(ctx) as *const _) };
                let mut seq = ctx_ref.sequence().unwrap();
                if let Some(bos) = model.bos_token() {
                    seq.push(bos);
                }
                (ctx_ref as *const Context, seq)
            },
            |(ctx_ptr, mut seq)| {
                let token = seq
                    .logits()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as i32)
                    .unwrap();
                seq.push(black_box(token));
                // Drop seq first (releases the borrow on ctx), then ctx
                drop(seq);
                unsafe { drop(Box::from_raw(ctx_ptr as *mut Context)) };
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_generate_10_tokens(c: &mut Criterion) {
    let model = load_model();
    let ctx_params = make_ctx_params();
    let prompt_tokens = model.tokenize("Once upon a time", true, false);

    c.bench_function("generate_10_tokens", |b| {
        b.iter_batched(
            || {
                let ctx = Box::new(Context::new(&model, &ctx_params).unwrap());
                let ctx_ref: &'static Context<'static> =
                    unsafe { &*(Box::into_raw(ctx) as *const _) };
                let mut seq = ctx_ref.sequence().unwrap();
                seq.extend(&prompt_tokens);
                (ctx_ref as *const Context, seq)
            },
            |(ctx_ptr, mut seq)| {
                for _ in 0..10 {
                    let token = seq
                        .logits()
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(i, _)| i as i32)
                        .unwrap();
                    if model.is_eog(token) {
                        break;
                    }
                    seq.push(black_box(token));
                }
                drop(seq);
                unsafe { drop(Box::from_raw(ctx_ptr as *mut Context)) };
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_tokenize, bench_token_to_piece, bench_decode_single_token, bench_generate_10_tokens);
criterion_main!(benches);
