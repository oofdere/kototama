//! Coverage for `Sequence::decode()` — the re-decode helper added in
//! commit `baec429` ("native sampler wip").
//!
//! `decode()` is documented as:
//!
//! > Re-decode the last token to refresh logits without pushing a new one.
//! > Useful after `pop()`, `remove()`, or other mutations that invalidate logits.
//!
//! It is the only public way to repopulate `Sequence`'s cached logits
//! short of pushing a fresh token. If it does not work, callers who
//! invalidated the cache (e.g. via `pop()`) either have to throw away a
//! token to get logits back or are stuck without logits.
//!
//! Despite that, no existing test exercises `decode()`:
//!   - PR #119 (`tests/sequence_logits_lifecycle.rs`) asserts that mutating
//!     ops *clear* the cache, never that `decode()` can bring it back.
//!   - PRs #122 / #113 cover the `Sampler` trait pipeline and pure-Rust
//!     `Temperature` / `MinP` / `Dist` math, neither of which touches
//!     `decode()`.
//!   - Closed PRs #75, #87, #90 predate the `decode()` method.
//!
//! Writing these tests surfaced a bug in `decode()` itself. The
//! implementation reuses the position of the existing last token:
//!
//! ```ignore
//! let pos = (self.tokens.len() - 1) as i32;
//! ... push_token(last_token, pos, self.id) ...
//! ```
//!
//! But `pop()` only removes the KV entry for the **popped** position, so
//! after `pop()` the KV cache still holds an entry at `tokens.len() - 1`.
//! `llama_decode` then rejects the batch with:
//!
//! > the tokens of sequence 0 in the input batch have inconsistent
//! > sequence positions ... required that the sequence positions remain
//! > consecutive: Y = X + 1
//!
//! …which `push_token` reports as `DecodeError::InvalidInput` and
//! `decode()` `unwrap_or_else`-panics on. The `#[should_panic]` test
//! below pins that bug. The `#[ignore]` tests below document the
//! intended contract so that once the bug is fixed (e.g. by
//! `kv_remove(pos..pos+1)` before re-decoding), running with
//! `--ignored` will verify it.

mod common;

use rusty_llama::Context;

// ---------- Empty-sequence contract (works today) ----------

#[test]
fn decode_on_empty_sequence_is_noop() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    assert!(seq.is_empty(), "precondition: fresh sequence is empty");
    assert!(seq.logits().is_none(), "precondition: no logits cached");

    // Must not panic, must not invent logits out of thin air.
    seq.decode();

    assert!(seq.is_empty(), "decode() must not add tokens");
    assert!(
        seq.logits().is_none(),
        "decode() on an empty sequence must not synthesise logits"
    );
}

// ---------- Bug pin: decode() panics in its documented use case ----------

#[test]
#[should_panic(expected = "decode failed")]
fn decode_after_pop_currently_panics() {
    // This pins the bug described in the module comment: `decode()` is
    // documented as useful after `pop()`, but the KV cache still holds an
    // entry at `tokens.len() - 1` (pop only freed the popped position), so
    // re-decoding at that position trips llama.cpp's consecutive-position
    // check. When `decode()` is fixed, this test will start failing and
    // should be deleted in favour of the `#[ignore]`d contract tests
    // below.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(
        tokens.len() >= 2,
        "test prompt must tokenize to >=2 tokens to allow a pop()"
    );
    seq.extend(&tokens);
    seq.pop();
    assert!(seq.logits().is_none());

    seq.decode(); // panics with "decode failed: InvalidInput"
}

// ---------- Intended-contract tests (ignored until the bug above is fixed) ----------

#[test]
#[ignore = "blocked by decode() bug pinned by decode_after_pop_currently_panics"]
fn decode_repopulates_logits_after_pop_invalidates_them() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    assert!(tokens.len() >= 2);
    seq.extend(&tokens);
    assert!(seq.logits().is_some(), "extend populates cache");

    seq.pop();
    assert!(
        seq.logits().is_none(),
        "precondition: pop() invalidates the cache"
    );

    seq.decode();
    let cached = seq
        .logits()
        .expect("decode() must repopulate logits after invalidation");
    assert_eq!(
        cached.len(),
        model.n_tokens() as usize,
        "repopulated logits must span the full vocab"
    );
}

#[test]
#[ignore = "blocked by decode() bug pinned by decode_after_pop_currently_panics"]
fn decode_does_not_change_tokens_or_length() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.pop();
    let len_before = seq.len();
    let tokens_before: Vec<i32> = seq.tokens().to_vec();

    seq.decode();

    assert_eq!(
        seq.len(),
        len_before,
        "decode() must not change the token count"
    );
    assert_eq!(
        seq.tokens(),
        tokens_before.as_slice(),
        "decode() must not mutate the token vector"
    );
}

#[test]
#[ignore = "blocked by decode() bug pinned by decode_after_pop_currently_panics"]
fn decode_does_not_change_pos_min_max() {
    // `decode()` is meant to re-run the last decode step against the same
    // KV-cache state, so the position range must not change. If it grew,
    // `decode()` would secretly be pushing a *new* token rather than
    // refreshing the existing last one.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.pop();
    let pos_min_before = seq.pos_min();
    let pos_max_before = seq.pos_max();

    seq.decode();

    assert_eq!(
        seq.pos_min(),
        pos_min_before,
        "decode() must not change pos_min"
    );
    assert_eq!(
        seq.pos_max(),
        pos_max_before,
        "decode() must not change pos_max"
    );
}

#[test]
#[ignore = "blocked by decode() bug pinned by decode_after_pop_currently_panics"]
fn decode_after_pop_matches_extend_to_same_length() {
    // Two paths to the same observable state must yield the same cached
    // logits:
    //   A) extend(&tokens[..n-1])      — natural push of n-1 tokens
    //   B) extend(&tokens); pop(); decode() — push n, drop last, re-decode
    // If `decode()` faithfully "re-runs the last decode step", A and B
    // must produce bit-identical logits.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let tokens = model.tokenize("once upon a time", false, false);
    assert!(tokens.len() >= 2);

    let mut seq_a = ctx.sequence().unwrap();
    seq_a.extend(&tokens[..tokens.len() - 1]);
    let logits_a: Vec<f32> = seq_a.logits().unwrap().to_vec();

    let mut seq_b = ctx.sequence().unwrap();
    seq_b.extend(&tokens);
    seq_b.pop();
    seq_b.decode();
    let logits_b: Vec<f32> = seq_b.logits().unwrap().to_vec();

    assert_eq!(
        logits_a, logits_b,
        "decode() after pop() must reproduce the natural push's logits"
    );
}

#[test]
#[ignore = "blocked by decode() bug pinned by decode_after_pop_currently_panics"]
fn repeated_decode_is_idempotent_on_logits() {
    // Calling decode() twice in a row replays the same single-token
    // decode step against the same prior KV-cache state, so the second
    // call's logits must equal the first call's.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();

    let tokens = model.tokenize("hello world", false, false);
    seq.extend(&tokens);
    seq.pop();

    seq.decode();
    let first: Vec<f32> = seq.logits().unwrap().to_vec();
    seq.decode();
    let second: Vec<f32> = seq.logits().unwrap().to_vec();

    assert_eq!(
        first, second,
        "successive decode() calls on an unchanged sequence must yield identical logits"
    );
}
