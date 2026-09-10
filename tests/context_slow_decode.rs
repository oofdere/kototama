mod common;

use rusty_llama::{Context, Model, Sequence};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Longer than the actor crate's default request timeout (5s).
const SLOW_DECODE: Duration = Duration::from_secs(7);

/// llama.cpp calls this from the compute threads. `data` points to a per-context
/// flag so that only the first decode of each context is delayed.
unsafe extern "C" fn slow_abort_callback(data: *mut c_void) -> bool {
    let slept = unsafe { &*(data as *const AtomicBool) };
    if !slept.swap(true, Ordering::SeqCst) {
        std::thread::sleep(SLOW_DECODE);
    }
    false
}

/// Context whose first decode takes `SLOW_DECODE`.
fn slow_sequence() -> (Model, Context, Sequence) {
    let (model, mut params) = common::load_model_and_context();
    let slept: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    params.abort_callback = Some(slow_abort_callback);
    params.abort_callback_data = slept as *const AtomicBool as *mut c_void;
    let ctx = Context::new(&model, &params).expect("failed to create context");
    let seq = ctx.sequence().expect("failed to acquire sequence");
    (model, ctx, seq)
}

#[test]
fn push_waits_for_a_decode_slower_than_the_request_timeout() {
    let (model, _ctx, mut seq) = slow_sequence();
    let token = model.bos_token().unwrap_or(0);

    let start = Instant::now();
    seq.push(token);

    assert!(
        start.elapsed() >= SLOW_DECODE,
        "the slow decode should have been waited for"
    );
    assert_eq!(seq.tokens(), &[token]);
    assert!(seq.logits().is_some(), "logits should be cached after push");
    assert_eq!(seq.pos_max(), 0);
}

#[test]
fn sequence_stays_usable_after_a_slow_decode() {
    let (model, _ctx, mut seq) = slow_sequence();
    let token = model.bos_token().unwrap_or(0);

    seq.push(token);
    seq.push(token);

    assert_eq!(seq.len(), 2);
    assert_eq!(seq.pos_min(), 0);
    assert_eq!(seq.pos_max(), 1);
}
