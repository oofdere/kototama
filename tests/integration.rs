mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};

// ---------- Model error handling ----------

#[test]
fn load_model_invalid_path_returns_err() {
    use rusty_llama::{Model, ModelParams};
    let params = ModelParams::new();
    let result = Model::load_from_file("/nonexistent/path/to/model.gguf", params);
    assert!(result.is_err());
}

// ---------- Tokenization edge cases ----------

#[test]
fn tokenize_empty_with_bos_gives_bos() {
    let model = common::load_model();
    let tokens = model.tokenize("", true, false);
    assert_eq!(tokens.len(), 1, "empty text with add_bos should produce exactly BOS");
    assert_eq!(tokens[0], model.bos_token().unwrap());
}

#[test]
fn tokenize_deterministic() {
    let model = common::load_model();
    let a = model.tokenize("the cat sat on the mat", false, false);
    let b = model.tokenize("the cat sat on the mat", false, false);
    assert_eq!(a, b, "tokenization should be deterministic");
}

// ---------- Sampler integration (requires model + context) ----------

#[test]
fn greedy_sample_matches_argmax() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("Once upon a time", true, false);
    seq.extend(&tokens);

    let mut argmax = 0i32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in seq.logits().unwrap().iter().enumerate() {
        if v > best {
            best = v;
            argmax = i as i32;
        }
    }

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());
    let sampled = seq.sample(&chain);

    assert_eq!(sampled, argmax, "greedy sampler should pick the argmax token");
}

#[test]
fn sample_with_temperature_does_not_crash() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", true, false);
    seq.extend(&tokens);

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::temp(0.8))
        .add(Sampler::top_k(40))
        .add(Sampler::top_p(0.95, 1))
        .add(Sampler::dist(42));
    let token = seq.sample(&chain);
    assert!(token >= 0 && token < model.n_tokens());
}

// ---------- Multi-sequence ----------

#[test]
fn two_sequences_independent() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq_a = ctx.sequence().unwrap();
    let mut seq_b = ctx.sequence().unwrap();

    let tokens_a = model.tokenize("hello", true, false);
    let tokens_b = model.tokenize("world", true, false);
    seq_a.extend(&tokens_a);
    seq_b.extend(&tokens_b);

    assert_eq!(seq_a.tokens(), tokens_a.as_slice());
    assert_eq!(seq_b.tokens(), tokens_b.as_slice());
    assert_ne!(seq_a.logits().unwrap(), seq_b.logits().unwrap(), "different prompts should produce different logits");
}

// ---------- Sequence logits state ----------

#[test]
fn logits_empty_before_push() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let seq = ctx.sequence().unwrap();
    assert!(seq.logits().is_none(), "logits should be None before any token is pushed");
}

// ---------- Generation determinism ----------

#[test]
fn greedy_generation_is_deterministic() {
    let (model, params) = common::load_model_and_context();

    let generate = || {
        let ctx = Context::new(&model, &params).unwrap();
        let mut seq = ctx.sequence().unwrap();
        let prompt = model.tokenize("the", true, false);
        seq.extend(&prompt);

        let mut tokens = Vec::new();
        for _ in 0..5 {
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
            tokens.push(token);
            seq.push(token);
        }
        tokens
    };

    let run1 = generate();
    let run2 = generate();
    assert_eq!(run1, run2, "greedy generation should be deterministic across runs");
}

// ---------- Token-to-piece coverage ----------

#[test]
fn most_vocab_tokens_have_pieces() {
    let model = common::load_model();
    let n = model.n_tokens();
    let mut ok = 0;
    for i in 0..n {
        if model.token_to_piece(i).is_ok() {
            ok += 1;
        }
    }
    assert!(
        ok as f64 / n as f64 > 0.99,
        "at least 99% of vocab tokens should decode to a piece, got {ok}/{n}"
    );
}
