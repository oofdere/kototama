//! Tests for the async façade: the `*_async` methods must behave exactly like
//! their synchronous counterparts. Driven with `block_on`, so no runtime.

use rusty_llama::asynchronous::block_on;
use rusty_llama::{Chain, Context, Dist, Greedy, Temperature, TopK};

mod common;

fn setup() -> (rusty_llama::Model, rusty_llama::ContextParams) {
    common::load_model_and_context()
}

#[test]
fn async_load_model_and_context() {
    let model = block_on(rusty_llama::Model::load_from_file_async(
        common::model_path().as_str(),
        rusty_llama::ModelParams::new(),
    ))
    .expect("async model load");
    let params = common::test_ctx_params();
    let ctx = block_on(Context::new_async(&model, &params)).expect("async context");
    assert_eq!(ctx.free_slots(), params.n_seq_max as usize);
}

#[test]
fn async_push_matches_sync_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let tokens = model.tokenize("hello world", false, false);

    let mut sync_seq = ctx.sequence().unwrap();
    sync_seq.extend(&tokens).unwrap();

    let mut async_seq = ctx.sequence().unwrap();
    block_on(async_seq.extend_async(&tokens)).unwrap();

    assert_eq!(async_seq.tokens(), sync_seq.tokens());
    assert_eq!(async_seq.logits(), sync_seq.logits(), "same tokens must give same logits");
}

#[test]
fn async_push_updates_len_and_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hi", false, false);

    block_on(seq.push_async(tokens[0])).unwrap();
    assert_eq!(seq.len(), 1);
    assert_eq!(seq.logits().unwrap().len(), model.n_tokens() as usize);
}

#[test]
fn async_pop_and_decode_refresh() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    block_on(seq.extend_async(&tokens)).unwrap();

    let popped = block_on(seq.pop_async());
    assert_eq!(popped, Some(tokens[tokens.len() - 1]));
    assert_eq!(seq.len(), tokens.len() - 1);
    assert!(seq.logits().is_none(), "pop invalidates cached logits");

    block_on(seq.decode_async()).unwrap();
    assert!(seq.logits().is_some(), "decode refreshes cached logits");
}

#[test]
fn async_pop_empty_returns_none() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_eq!(block_on(seq.pop_async()), None);
}

#[test]
fn async_remove_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", false, false);
    block_on(seq.extend_async(&tokens)).unwrap();

    assert!(block_on(seq.remove_async(0..1)));
    assert_eq!(seq.len(), tokens.len() - 1);
}

#[test]
fn async_decode_empty_is_noop() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    block_on(seq.decode_async()).unwrap();
    assert!(seq.logits().is_none());
}

#[test]
fn async_sampling_works_off_cached_logits() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon", false, false);
    block_on(seq.extend_async(&tokens)).unwrap();

    let argmax = seq
        .logits()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i32)
        .unwrap();

    let mut greedy = Greedy::new();
    assert_eq!(seq.sample(&mut greedy).unwrap(), argmax);
}

#[test]
fn async_greedy_generation_is_deterministic() {
    let (model, params) = setup();

    let generate = || {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        let prompt = model.tokenize("the", true, false);
        block_on(seq.extend_async(&prompt)).unwrap();

        let mut sampler = Chain::new().with(TopK::new(1)).with(Temperature::new(1.0));
        let mut tokens = Vec::new();
        for _ in 0..5 {
            let token = seq.sample(&mut sampler).unwrap();
            if model.is_eog(token) {
                break;
            }
            tokens.push(token);
            block_on(seq.push_async(token)).unwrap();
        }
        tokens
    };

    assert_eq!(generate(), generate(), "async greedy generation must be deterministic");
}

#[test]
fn async_futures_are_send() {
    // Compile-time check: the futures must be Send so any multi-threaded
    // executor can drive them.
    fn assert_send<T: Send>(_: T) {}

    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert_send(seq.push_async(1));
    assert_send(seq.extend_async(&[1, 2, 3]));
    assert_send(seq.pop_async());
    assert_send(seq.decode_async());
    assert_send(seq.remove_async(0..1));
}

#[test]
fn async_dist_sampling_still_random_but_valid() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", false, false);
    block_on(seq.extend_async(&tokens)).unwrap();

    let mut dist = Dist::new(7);
    let token = seq.sample(&mut dist).unwrap();
    assert!((0..model.n_tokens()).contains(&token));
}
