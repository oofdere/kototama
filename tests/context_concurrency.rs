// Concurrent use of a single `Context` from multiple threads.
//
// `lib.rs` advertises "Thread safety via an actor model (powered by Spawned)"
// and `Context` is `Clone + Send + Sync`, but every existing test runs on the
// main thread sequentially:
//
//   - `tests/context.rs::context_clone_*` only verify that two `Context`
//     handles see the same shared state, not that concurrent calls from
//     distinct threads are actually safe.
//   - `tests/backend.rs::*` exercises `Backend::acquire` / `Drop`. PR #65
//     adds a concurrent acquire/drop stress test, but that's `Backend`,
//     not `Context` / `Sequence`.
//
// The actor mailbox's whole job is to serialize cross-thread access to
// `llama_context`. A regression that bypassed the mailbox — or a `Sequence`
// that accidentally lost its `Send` bound — would not be caught by anything
// currently in the suite or any open test PR.
//
// These tests are in their own file so they don't conflict with the in-flight
// PRs that touch `tests/context.rs` (#43, #78, #84).
//
// NOTE: the suite-wide `--test-threads=1` flag in `AGENTS.md` serializes
// **tests**, not threads spawned **inside** a test. Spawning workers from
// within a test is fine — the threads share one already-initialized
// llama.cpp backend.

mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};
use std::sync::{Arc, Barrier};
use std::thread;

// ---------- Send/Sync compile-time witnesses ----------
// If `Context` ever loses `Send + Sync`, or `Sequence` loses `Send`, this
// file won't compile — making the regression a build-time failure rather
// than a silent loss of the documented thread-safety guarantee.

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn context_is_send_and_sync() {
    assert_send::<Context>();
    assert_sync::<Context>();
}

#[test]
fn sequence_is_send() {
    assert_send::<rusty_llama::Sequence>();
}

// ---------- Concurrent sequence checkout ----------

#[test]
fn concurrent_checkout_allocates_distinct_seq_ids() {
    // Spin up `n_seq_max` worker threads that all try to check out a
    // sequence at the same instant. Every successful checkout must succeed,
    // and pushing a distinct token through each must yield distinct
    // logits — proving each `Sequence` is talking to its own KV slot
    // rather than two `Sequence` handles aliasing the same slot.
    let (model, mut params) = common::load_model_and_context();
    params.n_seq_max = 8;
    let n = params.n_seq_max as usize;

    let ctx = Context::new(&model, &params).unwrap();
    let barrier = Arc::new(Barrier::new(n));

    // Each worker uses a distinct prompt so that aliased slots would
    // collapse their logits to a single value.
    let prompts: Vec<Vec<i32>> = (0..n)
        .map(|i| model.tokenize(&format!("prompt {i}"), false, false))
        .collect();
    assert!(prompts.iter().all(|p| !p.is_empty()));

    let handles: Vec<_> = prompts
        .into_iter()
        .enumerate()
        .map(|(i, tokens)| {
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                let mut seq = ctx.sequence().expect("slot available");
                barrier.wait();
                seq.extend(&tokens);
                let last = seq.logits().unwrap().last().copied().unwrap();
                (i, seq, last)
            })
        })
        .collect();

    let mut seqs = Vec::with_capacity(n);
    let mut last_logits = Vec::with_capacity(n);
    for h in handles {
        let (_, seq, last) = h.join().unwrap();
        seqs.push(seq);
        last_logits.push(last);
    }
    assert_eq!(seqs.len(), n);
    assert_eq!(ctx.free_slots(), 0, "all slots should be in use");

    // Sanity: the prompts differ, so at least two logits values must too.
    // (TinyStories is small — we don't require *all* to differ, just that
    // the per-sequence state isn't collapsed onto one shared slot.)
    let distinct: std::collections::HashSet<u32> =
        last_logits.iter().map(|f| f.to_bits()).collect();
    assert!(
        distinct.len() > 1,
        "distinct prompts on distinct slots should yield more than one logit value, got {last_logits:?}"
    );

    drop(seqs);
    assert_eq!(ctx.free_slots(), n, "all slots should be returned");
}

#[test]
fn concurrent_checkout_does_not_over_allocate() {
    // 2 * n_seq_max threads race for n_seq_max slots. Exactly n_seq_max
    // checkouts must succeed and exactly n_seq_max must return None — no
    // double-allocation, no missed allocations.
    let (model, mut params) = common::load_model_and_context();
    params.n_seq_max = 4;
    let n_slots = params.n_seq_max as usize;
    let n_workers = n_slots * 2;

    let ctx = Context::new(&model, &params).unwrap();
    let barrier = Arc::new(Barrier::new(n_workers));

    let handles: Vec<_> = (0..n_workers)
        .map(|_| {
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                ctx.sequence()
            })
        })
        .collect();

    let mut some_count = 0;
    let mut none_count = 0;
    let mut seqs = Vec::new();
    for h in handles {
        match h.join().unwrap() {
            Some(s) => {
                some_count += 1;
                seqs.push(s);
            }
            None => none_count += 1,
        }
    }

    assert_eq!(some_count, n_slots, "exactly n_seq_max checkouts should succeed");
    assert_eq!(none_count, n_workers - n_slots, "the rest must observe exhaustion");
    assert_eq!(ctx.free_slots(), 0);

    drop(seqs);
    assert_eq!(ctx.free_slots(), n_slots);
}

// ---------- Concurrent push / decode ----------

#[test]
fn concurrent_push_on_independent_sequences() {
    // Each worker owns its own `Sequence` and pushes a distinct prompt's
    // worth of tokens through the actor concurrently. The actor mailbox
    // must serialize the underlying `llama_decode` calls without losing
    // or interleaving tokens between sequences.
    let (model, mut params) = common::load_model_and_context();
    params.n_seq_max = 4;
    let ctx = Context::new(&model, &params).unwrap();

    // Pre-tokenize on the main thread so the workers only exercise the
    // actor path.
    let prompts: Vec<Vec<i32>> = ["hello", "world", "once upon", "the cat"]
        .iter()
        .map(|p| model.tokenize(p, false, false))
        .collect();
    assert!(prompts.iter().all(|p| !p.is_empty()));

    let barrier = Arc::new(Barrier::new(prompts.len()));

    let handles: Vec<_> = prompts
        .into_iter()
        .map(|tokens| {
            let ctx = ctx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                let mut seq = ctx.sequence().expect("slot available");
                barrier.wait();
                for &t in &tokens {
                    seq.push(t);
                }
                // Return tokens + logits length so the main thread can verify.
                let logits_len = seq.logits().map(|l| l.len()).unwrap_or(0);
                (tokens, seq.tokens().to_vec(), logits_len)
            })
        })
        .collect();

    let vocab = model.n_tokens() as usize;
    for h in handles {
        let (expected, got, logits_len) = h.join().unwrap();
        assert_eq!(got, expected, "sequence should hold exactly its own pushes");
        assert_eq!(logits_len, vocab, "logits cache should be vocab-sized after push");
    }

    assert_eq!(ctx.free_slots(), params.n_seq_max as usize);
}

// ---------- Concurrent sample ----------

#[test]
fn concurrent_greedy_sample_matches_serial() {
    // Greedy sampling is deterministic: for a fixed prompt the sampled
    // token is the argmax of `seq.logits()`. Running the sample step
    // concurrently across workers must produce the same token as the
    // single-threaded reference, because the actor serializes the
    // `llama_sampler_sample` call against the per-context state.
    let (model, mut params) = common::load_model_and_context();
    params.n_seq_max = 4;
    let ctx = Context::new(&model, &params).unwrap();

    let prompt = model.tokenize("Once upon a time", false, false);
    assert!(!prompt.is_empty());

    // Reference: serial computation of the greedy token.
    let reference = {
        let mut seq = ctx.sequence().unwrap();
        seq.extend(&prompt);
        let chain = SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
        seq.sample(&chain)
    };
    assert!(reference >= 0 && reference < model.n_tokens());

    // Concurrent: four workers each independently set up the same prompt
    // and sample. All four must agree with the reference.
    let n_workers = params.n_seq_max as usize;
    let barrier = Arc::new(Barrier::new(n_workers));
    let handles: Vec<_> = (0..n_workers)
        .map(|_| {
            let ctx = ctx.clone();
            let prompt = prompt.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                let mut seq = ctx.sequence().expect("slot available");
                seq.extend(&prompt);
                let chain =
                    SamplerChain::new(&SamplerChainParams::new()).add(Sampler::greedy());
                barrier.wait();
                seq.sample(&chain)
            })
        })
        .collect();

    for h in handles {
        let got = h.join().unwrap();
        assert_eq!(
            got, reference,
            "greedy sample must be deterministic across threads"
        );
    }
}

// ---------- Drop ordering across threads ----------

#[test]
fn sequence_drop_on_worker_thread_returns_slot() {
    // A `Sequence` checked out on the main thread and moved into a worker
    // must release its slot on `Drop` — the `ReleaseSeq` message has to
    // reach the actor regardless of which thread runs the destructor.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();

    let initial = ctx.free_slots();
    let seq = ctx.sequence().unwrap();
    assert_eq!(ctx.free_slots(), initial - 1);

    let handle = thread::spawn(move || {
        // Sequence is moved into this thread; it is dropped here.
        drop(seq);
    });
    handle.join().unwrap();

    assert_eq!(
        ctx.free_slots(),
        initial,
        "slot must be released even when the Sequence is dropped off-thread"
    );
}

#[test]
fn context_handle_outlives_creator_thread() {
    // The `Context` is `Arc<ContextInner>`; the actor thread is owned by
    // the inner. A `Context` cloned into a worker thread must remain
    // usable after the original handle is dropped on the main thread,
    // because the actor stays alive as long as any clone exists.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let ctx_clone = ctx.clone();
    drop(ctx);

    let handle = thread::spawn(move || {
        // Worker still has a live handle; the actor must still answer.
        let seq = ctx_clone.sequence().expect("actor still alive");
        let n = ctx_clone.n_ctx();
        assert!(seq.is_empty());
        n
    });

    let n_ctx = handle.join().unwrap();
    assert!(n_ctx > 0);
}
