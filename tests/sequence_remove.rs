mod common;

use rusty_llama::Context;

fn seq_of(n: usize) -> (rusty_llama::Model, Context, rusty_llama::Sequence) {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
    seq.extend(&tokens);
    (model, ctx, seq)
}

#[test]
fn remove_middle_rebases_kv_positions() {
    let (_model, _ctx, mut seq) = seq_of(5);
    assert!(seq.remove(1..3));
    assert_eq!(seq.len(), 3);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 2);
}

#[test]
fn push_after_middle_remove_still_works() {
    let (_model, _ctx, mut seq) = seq_of(5);
    assert!(seq.remove(1..3));
    seq.push(7);
    assert_eq!(seq.len(), 4);
    assert_eq!(seq.pos_max(), 3);
    assert!(seq.logits().is_some());
}

#[test]
fn remove_prefix_rebases_kv_positions() {
    let (_model, _ctx, mut seq) = seq_of(4);
    assert!(seq.remove(0..2));
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 1);
    seq.push(9);
    assert_eq!(seq.pos_max(), 2);
}

#[test]
fn remove_suffix_leaves_positions_intact() {
    let (_model, _ctx, mut seq) = seq_of(4);
    assert!(seq.remove(2..4));
    assert_eq!(seq.len(), 2);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 1);
}

#[test]
fn remove_out_of_bounds_is_rejected() {
    let (_model, _ctx, mut seq) = seq_of(3);
    assert!(!seq.remove(1..99));
    assert_eq!(seq.len(), 3);
    assert_eq!(seq.pos_max(), 2);
    // the sequence is still usable
    seq.push(5);
    assert_eq!(seq.len(), 4);
}

#[test]
fn remove_inverted_range_is_rejected() {
    let (_model, _ctx, mut seq) = seq_of(3);
    assert!(!seq.remove(2..1));
    assert_eq!(seq.len(), 3);
    assert_eq!(seq.pos_max(), 2);
}

#[test]
fn remove_empty_range_is_a_noop() {
    let (_model, _ctx, mut seq) = seq_of(3);
    assert!(seq.remove(1..1));
    assert_eq!(seq.len(), 3);
    assert_eq!(seq.pos_max(), 2);
}

#[test]
fn remove_all_empties_the_sequence() {
    let (_model, _ctx, mut seq) = seq_of(3);
    assert!(seq.remove(0..3));
    assert_eq!(seq.len(), 0);
    assert!(seq.is_empty());
    seq.push(4);
    assert_eq!(seq.pos_max(), 0);
}
