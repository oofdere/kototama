mod common;

use rusty_llama::{Chain, Context, Dist, Greedy, Sampler, Temperature, Token};

/// Counts how many times the transform ran, and rotates the logits left by one
/// so that applying it twice is observably different from applying it once.
#[derive(Default)]
struct Rotate {
    applies: usize,
}

impl Sampler for Rotate {
    fn apply_mut(&mut self, logits: &mut [f32]) {
        self.applies += 1;
        logits.rotate_left(1);
    }
}

fn argmax(logits: &[f32]) -> Token {
    let (id, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    id as Token
}

fn seq_with_logits() -> (rusty_llama::Sequence, Vec<f32>) {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello world", true, false);
    seq.extend(&tokens);
    let logits = seq.logits().unwrap().to_vec();
    (seq, logits)
}

#[test]
fn sequence_sample_applies_transform_once() {
    let (seq, _) = seq_with_logits();
    let mut rot = Rotate::default();
    seq.sample(&mut rot).unwrap();
    assert_eq!(rot.applies, 1, "transform must run exactly once");
}

#[test]
fn sequence_sample_matches_direct_sample() {
    let (seq, logits) = seq_with_logits();

    let mut rot = Rotate::default();
    let from_seq = seq.sample(&mut rot).unwrap();

    let mut direct = Rotate::default();
    let from_logits = direct.sample(&logits);

    assert_eq!(from_seq, from_logits);
}

#[test]
fn sequence_sample_selects_from_transformed_logits() {
    let (seq, logits) = seq_with_logits();

    let mut rotated = logits.clone();
    rotated.rotate_left(1);
    let expected = argmax(&rotated);

    let mut rot = Rotate::default();
    assert_eq!(seq.sample(&mut rot).unwrap(), expected);
}

#[test]
fn sequence_sample_does_not_double_scale_temperature() {
    let (seq, logits) = seq_with_logits();

    // Temperature 0.5 doubles the logits; sampling through the sequence must
    // not scale them by 4x, so the Dist draw has to match the direct one.
    let mut chain = Chain::new()
        .with(Temperature::new(0.5))
        .with(Dist::new(1234));
    let from_seq = seq.sample(&mut chain).unwrap();

    let scaled: Vec<f32> = logits.iter().map(|l| l * 2.0).collect();
    let expected = Dist::new(1234).sample(&scaled);

    assert_eq!(from_seq, expected);
}

#[test]
fn chain_sample_applies_last_transform_once() {
    let mut chain = Chain::new().with(Rotate::default());
    // logits chosen so a single rotation yields argmax 1 and a double
    // rotation yields argmax 0.
    let token = chain.sample(&[0.0, 1.0, 2.0]);
    assert_eq!(token, 1);
}

#[test]
fn chain_sample_selector_last_is_unchanged() {
    let mut chain = Chain::new().with(Rotate::default()).with(Greedy::new());
    // [0.0, 1.0, 2.0] rotated once -> [1.0, 2.0, 0.0]
    assert_eq!(chain.sample(&[0.0, 1.0, 2.0]), 1);
}

#[test]
fn empty_chain_sample_is_argmax() {
    let mut chain = Chain::new();
    assert_eq!(chain.sample(&[0.0, 3.0, 1.0]), 1);
}
