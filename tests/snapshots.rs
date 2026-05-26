mod common;

use rusty_llama::Context;

// ---------- Tokenization snapshots ----------
// These snapshot the token ID vectors. They are stable as long as the model
// and llama.cpp tokenizer are unchanged. Run `cargo insta review` after
// bumping llama.cpp to accept updated snapshots.

#[test]
fn snapshot_tokenize_hello_world() {
    let model = common::load_model();
    let tokens = model.tokenize("Hello, world!", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_with_bos() {
    let model = common::load_model();
    let tokens = model.tokenize("Hello, world!", true, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_multiline() {
    let model = common::load_model();
    let tokens = model.tokenize("line one\nline two\nline three", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_numbers() {
    let model = common::load_model();
    let tokens = model.tokenize("1, 2, 3, 4, 5", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

// ---------- Generation snapshots ----------
// Greedy (argmax) sampling produces a deterministic token sequence for a given
// model and llama.cpp version. We snapshot the token IDs — not the decoded
// text — so the snapshot stays valid even if token_to_piece rendering changes.
// Run `cargo insta review` after bumping llama.cpp to accept updated snapshots.

fn greedy_generate(prompt: &str, n_tokens: usize) -> Vec<i32> {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let prompt_tokens = model.tokenize(prompt, true, false);
    seq.extend(&prompt_tokens);

    let mut generated = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        let token = seq
            .logits()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as i32)
            .unwrap();
        if model.is_eog(token) {
            break;
        }
        generated.push(token);
        seq.push(token);
    }
    generated
}

#[test]
fn snapshot_generate_10_tokens() {
    let tokens = greedy_generate("Once upon a time", 10);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_generate_numbers() {
    let tokens = greedy_generate("1, 2, 3,", 8);
    insta::assert_yaml_snapshot!(tokens);
}
