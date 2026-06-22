//! Coverage for `Dist::sample_mut` — a gap in the native `Sampler`
//! trait introduced in commit `baec429` ("native sampler wip").
//!
//! `Dist` overrides the trait's `sample(&[f32])` to do stochastic
//! softmax sampling, but **does not** override `sample_mut(&mut [f32])`.
//! `sample_mut` therefore falls through to the trait default, which is
//! `apply_mut` followed by an argmax. For `Dist` that's
//!
//! ```ignore
//! fn apply_mut(&self, _logits: &mut [f32]) {} // no-op
//! // trait default sample_mut: apply_mut, then argmax
//! ```
//!
//! …i.e. `Dist::sample_mut` is silently a deterministic greedy selector
//! — even though `Dist::sample` on the same logits is stochastic. The
//! two methods are otherwise documented as equivalent (see PR #110's
//! sampler module docs: "Same as sample but consumes the caller's
//! buffer instead of allocating a copy").
//!
//! Existing coverage does not catch this:
//!   - PR #113 (`tests/sampler_trait.rs`) tests `sample_mut` only for
//!     `Temperature` (where the trait default happens to be correct
//!     because the post-transform argmax is still the right answer).
//!   - PR #109 (`tests/sampler_nan.rs`) tests `sample_mut` only against
//!     NaN, not against `Dist` at all.
//!   - PR #122 (`tests/sequence_sample_native.rs`) exercises
//!     `Sequence::sample`, which internally calls `sampler.sample` (not
//!     `sample_mut`), so it never goes through this path.
//!
//! The tests below pin the current (buggy) behaviour so a future
//! refactor can't quietly break the contract further, and the
//! `#[ignore]`d tests document the intended behaviour that `Dist`
//! should override `sample_mut` to delegate to its stochastic `sample`.

use rusty_llama::{Dist, Sampler};

// ---------- Bug pin: Dist::sample_mut is deterministic argmax today ----------
//
// Note on argmax tie-breaking: `Iterator::max_by` keeps the *last*
// equally-maximum element, so on uniform logits the trait-default
// argmax returns `logits.len() - 1`. That makes the deterministic
// nature of today's `Dist::sample_mut` easy to spot.

#[test]
fn dist_sample_mut_currently_returns_last_argmax_on_uniform_logits() {
    let dist = Dist::new(42);
    let mut logits = vec![0.0_f32; 8];
    let token = dist.sample_mut(&mut logits);
    assert_eq!(
        token, 7,
        "Dist::sample_mut on uniform logits returns the last argmax (len-1) \
         because it falls through to the trait's argmax default"
    );
}

#[test]
fn dist_sample_mut_currently_ignores_seed_on_uniform_logits() {
    // If `sample_mut` truly delegated to `sample`, two `Dist`s with
    // different seeds would produce different streams on uniform
    // logits. The argmax fallback ignores the seed, so both return the
    // last-tied argmax every time.
    let a = Dist::new(1);
    let b = Dist::new(99_999);
    for _ in 0..50 {
        let mut la = vec![0.0_f32; 8];
        let mut lb = vec![0.0_f32; 8];
        assert_eq!(a.sample_mut(&mut la), 7);
        assert_eq!(b.sample_mut(&mut lb), 7);
    }
}

#[test]
fn dist_sample_mut_currently_picks_argmax_when_one_dominates() {
    // With one logit clearly the largest, argmax happens to coincide
    // with the (nearly certain) stochastic pick. That's why
    // `Dist::sample_mut` looks correct in casual use and the gap has
    // gone unnoticed. Pin this so a future "fix" that makes
    // `sample_mut` stochastic doesn't silently change the result here
    // either — under that fix this case must still return index 1
    // with overwhelming probability.
    let dist = Dist::new(7);
    let mut logits = vec![0.0, 14.0, 0.0, 0.0];
    let token = dist.sample_mut(&mut logits);
    assert_eq!(token, 1);
}

#[test]
fn dist_sample_mut_does_not_advance_rng_today() {
    // The trait default `sample_mut` never touches `Dist`'s RNG, so two
    // back-to-back calls observe an unchanged stream. After a fix that
    // routes `sample_mut` through `Dist::sample`, this assertion will
    // become a false statement (because `sample` does advance the RNG),
    // and the test will need to flip — that flip is the signal that the
    // gap has been closed.
    let dist = Dist::new(42);
    // Use uniform logits so any "real" sampling would be visible as
    // different draws; today's `sample_mut` returns the last argmax
    // both times.
    let mut a = vec![0.0_f32; 8];
    let mut b = vec![0.0_f32; 8];
    assert_eq!(dist.sample_mut(&mut a), 7);
    assert_eq!(dist.sample_mut(&mut b), 7);
}

// ---------- Intended contract (ignored until Dist::sample_mut is overridden) ----------

#[test]
#[ignore = "blocked: Dist does not override Sampler::sample_mut; falls back to argmax"]
fn dist_sample_mut_should_be_stochastic_on_uniform_logits() {
    // The intended contract: `sample_mut` is `sample` but in-place, so
    // on uniform logits it must explore the full index space across
    // many draws — not be stuck on the last-tied argmax (index 7).
    let dist = Dist::new(42);
    let mut seen_other = false;
    for _ in 0..200 {
        let mut logits = vec![0.0_f32; 8];
        if dist.sample_mut(&mut logits) != 7 {
            seen_other = true;
            break;
        }
    }
    assert!(
        seen_other,
        "Dist::sample_mut on uniform logits must be stochastic, \
         not stuck on the trait-default argmax"
    );
}

#[test]
#[ignore = "blocked: Dist does not override Sampler::sample_mut; falls back to argmax"]
fn dist_sample_mut_should_match_dist_sample_with_same_seed() {
    // Two `Dist`s seeded identically should produce identical streams
    // regardless of which entry-point (`sample` or `sample_mut`) the
    // caller uses. Today this fails on any logits where the stochastic
    // pick differs from the argmax (e.g. uniform).
    let a = Dist::new(123);
    let b = Dist::new(123);
    let logits = vec![0.0_f32; 8];
    for _ in 0..32 {
        let from_sample = a.sample(&logits);
        let mut buf = logits.clone();
        let from_sample_mut = b.sample_mut(&mut buf);
        assert_eq!(from_sample, from_sample_mut);
    }
}

// ---------- Future-proofing contract: passes today *and* after the fix ----------

#[test]
fn dist_sample_mut_does_not_mutate_logits() {
    // `Dist::apply_mut` is documented as a no-op, so `sample_mut`
    // (which is `apply_mut` then select) must leave the buffer
    // unchanged. The trait-default `sample_mut` honours this today only
    // because `apply_mut` itself is a no-op; once `sample_mut` is
    // overridden to delegate to `sample`, the no-mutation contract
    // still has to hold. Asserting it now means a future fix can't
    // regress it.
    let dist = Dist::new(5);
    let original = vec![1.0_f32, 2.0, 3.0, 0.5];
    let mut buf = original.clone();
    let _ = dist.sample_mut(&mut buf);
    assert_eq!(buf, original);
}
