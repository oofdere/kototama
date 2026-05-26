use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use rusty_llama::{Context, ContextParams};

fn load_model() -> rusty_llama::Model {
    rusty_llama::test_common::load_model()
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
                let ctx = Context::new(&model, &ctx_params).unwrap();
                let mut seq = ctx.sequence().unwrap();
                let seed_token = model.bos_token().unwrap_or(1);
                seq.push(seed_token);
                (ctx, seq)
            },
            |(_ctx, mut seq)| {
                let token = seq
                    .logits()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as i32)
                    .unwrap();
                seq.push(black_box(token));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_generate_10_tokens(c: &mut Criterion) {
    let model = load_model();
    let ctx_params = make_ctx_params();
    let prompt_tokens = model.tokenize("Once upon a time", true, false);
    assert!(!prompt_tokens.is_empty(), "tokenize must produce at least one token with add_bos=true");

    c.bench_function("generate_10_tokens", |b| {
        b.iter_batched(
            || {
                let ctx = Context::new(&model, &ctx_params).unwrap();
                let mut seq = ctx.sequence().unwrap();
                seq.extend(&prompt_tokens);
                (ctx, seq)
            },
            |(_ctx, mut seq)| {
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
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_context_creation(c: &mut Criterion) {
    let model = load_model();
    let ctx_params = make_ctx_params();

    c.bench_function("context_creation", |b| {
        b.iter(|| Context::new(black_box(&model), black_box(&ctx_params)).unwrap())
    });
}

fn bench_sequence_extend(c: &mut Criterion) {
    let model = load_model();
    let ctx_params = make_ctx_params();
    let tokens: Vec<i32> = (0..100).map(|_| model.bos_token().unwrap_or(1)).collect();

    c.bench_function("sequence_extend_100_tokens", |b| {
        b.iter_batched(
            || {
                let ctx = Context::new(&model, &ctx_params).unwrap();
                let seq = ctx.sequence().unwrap();
                (ctx, seq)
            },
            |(_ctx, mut seq)| {
                seq.extend(black_box(&tokens));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_tokenize,
    bench_token_to_piece,
    bench_decode_single_token,
    bench_generate_10_tokens,
    bench_context_creation,
    bench_sequence_extend,
);
criterion_main!(benches);
