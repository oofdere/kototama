mod common;

use rusty_llama::Context;

// ---------- Tokenization snapshots ----------
// These snapshot the token ID vectors. They are stable as long as the model
// and llama.cpp tokenizer are unchanged. Run `cargo insta review` after
// bumping llama.cpp to accept updated snapshots.

#[test]
fn snapshot_tokenize_hello_world() {
    let Some(model) = common::try_load_model() else { return };
    let tokens = model.tokenize("Hello, world!", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_with_bos() {
    let Some(model) = common::try_load_model() else { return };
    let tokens = model.tokenize("Hello, world!", true, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_multiline() {
    let Some(model) = common::try_load_model() else { return };
    let tokens = model.tokenize("line one\nline two\nline three", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_tokenize_numbers() {
    let Some(model) = common::try_load_model() else { return };
    let tokens = model.tokenize("1 + 1 = 2", false, false);
    insta::assert_yaml_snapshot!(tokens);
}

// ---------- Generation snapshots ----------
// Greedy (argmax) sampling produces a deterministic token sequence for a given
// model and llama.cpp version. We snapshot the token IDs — not the decoded
// text — so the snapshot stays valid even if token_to_piece rendering changes.
// Run `cargo insta review` after bumping llama.cpp to accept updated snapshots.

fn greedy_generate(prompt: &str, n_tokens: usize) -> Vec<i32> {
    let Some((model, params)) = common::try_load_model_and_context() else {
        return vec![];
    };
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let prompt_tokens = model.tokenize(prompt, true, false);
    seq.extend(&prompt_tokens);

    let mut generated = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
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
        generated.push(token);
        seq.push(token);
    }
    generated
}

#[test]
fn snapshot_generate_10_tokens() {
    let tokens = greedy_generate("Once upon a time", 10);
    if tokens.is_empty() {
        return; // model not available, skip
    }
    insta::assert_yaml_snapshot!(tokens);
}

#[test]
fn snapshot_generate_numbers() {
    let tokens = greedy_generate("1, 2, 3,", 8);
    if tokens.is_empty() {
        return;
    }
    insta::assert_yaml_snapshot!(tokens);
}
