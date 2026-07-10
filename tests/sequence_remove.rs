use rusty_llama::{test_common, Context};

#[test]
fn remove_compacts_remaining_kv_positions() {
    let (model, params) = test_common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("once upon a time", false, false);
    assert!(tokens.len() >= 4);
    seq.extend(&tokens);

    assert!(seq.remove(1..3));

    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), (seq.len() - 1) as i32);

    let next = model.tokenize("!", false, false);
    seq.push(next[0]);
    assert_eq!(seq.pos_max(), (seq.len() - 1) as i32);
}
