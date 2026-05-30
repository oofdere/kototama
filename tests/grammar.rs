mod common;

use rusty_llama::{Context, Sampler, SamplerChain, SamplerChainParams};
use std::ffi::CString;

// Get a raw `*const llama_vocab` from a Model via the public llama_sys API.
// The Model's internal `vocab_ptr()` is pub(crate); for test crates outside
// the wrapper we round-trip through the C API instead.
fn vocab_ptr_of(model: &rusty_llama::Model) -> *const llama_sys::llama_vocab {
    unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) }
}

// ---------- Sampler::grammar ----------

#[test]
fn grammar_init_simple_alternation() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"yes\" | \"no\"").unwrap();
    let root = CString::new("root").unwrap();
    // SAFETY: grammar/root C strings live for the duration of the FFI call.
    // llama_sampler_init_grammar parses and copies them internally.
    let _s = Sampler::grammar(vocab, grammar.as_ptr(), root.as_ptr());
}

#[test]
fn grammar_init_empty_is_unconstrained() {
    // Per the upstream API, an empty grammar string yields an unconstrained
    // grammar (rather than a null sampler). Exercise that path.
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("").unwrap();
    let root = CString::new("root").unwrap();
    let _s = Sampler::grammar(vocab, grammar.as_ptr(), root.as_ptr());
}

#[test]
fn grammar_init_charset() {
    // Slightly richer grammar that uses a character-class rule. Both the
    // production and the start-symbol resolution paths get exercised here.
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= [a-z]+").unwrap();
    let root = CString::new("root").unwrap();
    let _s = Sampler::grammar(vocab, grammar.as_ptr(), root.as_ptr());
}

#[test]
fn grammar_in_chain_samples_valid_token() {
    // End-to-end: build a chain ending in a grammar sampler + greedy and verify
    // sampling returns an in-vocab token. The grammar is unconstrained so any
    // token from the model is acceptable; the goal is to confirm the FFI
    // pipeline (chain.add(grammar), sample on a live context) doesn't crash
    // or produce out-of-range token ids.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", true, false);
    seq.extend(&tokens);

    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("").unwrap();
    let root = CString::new("root").unwrap();

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::grammar(vocab, grammar.as_ptr(), root.as_ptr()))
        .add(Sampler::greedy());
    let token = seq.sample(&chain);
    assert!(
        token >= 0 && token < model.n_tokens(),
        "grammar-constrained sample should be in vocab range, got {token}"
    );
}

// ---------- Sampler::grammar_lazy ----------

#[test]
fn grammar_lazy_init_no_triggers() {
    // num_trigger_words = 0 and num_trigger_tokens = 0 means the pointer
    // arguments are unused, so null is accepted.
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();
    let _s = Sampler::grammar_lazy(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        0,
    );
}

#[test]
fn grammar_lazy_init_with_trigger_word() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();

    let trigger = CString::new("BEGIN").unwrap();
    // The C API takes `*mut *const c_char` — an array of C string pointers.
    let mut trigger_ptrs: Vec<*const std::os::raw::c_char> = vec![trigger.as_ptr()];

    let _s = Sampler::grammar_lazy(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        trigger_ptrs.as_mut_ptr(),
        trigger_ptrs.len(),
        std::ptr::null(),
        0,
    );
    // trigger and trigger_ptrs must outlive the constructor call above;
    // by being bound to locals here they live to the end of the test.
    drop(trigger_ptrs);
    drop(trigger);
}

#[test]
fn grammar_lazy_init_with_trigger_token() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();

    let bos = model.bos_token().unwrap_or(1);
    let trigger_tokens: Vec<i32> = vec![bos];

    let _s = Sampler::grammar_lazy(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        trigger_tokens.as_ptr(),
        trigger_tokens.len(),
    );
}

// ---------- Sampler::grammar_lazy_patterns ----------

#[test]
fn grammar_lazy_patterns_init_no_triggers() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();
    let _s = Sampler::grammar_lazy_patterns(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        0,
    );
}

#[test]
fn grammar_lazy_patterns_init_with_pattern() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();

    // A capturing pattern: upstream uses the first capture group as the grammar
    // input once the pattern matches.
    let pattern = CString::new("^BEGIN:(.*)").unwrap();
    let mut pattern_ptrs: Vec<*const std::os::raw::c_char> = vec![pattern.as_ptr()];

    let _s = Sampler::grammar_lazy_patterns(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        pattern_ptrs.as_mut_ptr(),
        pattern_ptrs.len(),
        std::ptr::null(),
        0,
    );
    drop(pattern_ptrs);
    drop(pattern);
}

#[test]
fn grammar_lazy_patterns_init_with_trigger_token() {
    let model = common::load_model();
    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();

    let bos = model.bos_token().unwrap_or(1);
    let trigger_tokens: Vec<i32> = vec![bos];

    let _s = Sampler::grammar_lazy_patterns(
        vocab,
        grammar.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        trigger_tokens.as_ptr(),
        trigger_tokens.len(),
    );
}

// ---------- End-to-end sampling through a lazy grammar ----------

#[test]
fn grammar_lazy_in_chain_samples_valid_token() {
    // Lazy grammar with no triggers stays inactive, so it should be a no-op
    // in the chain — but the FFI plumbing still has to handle the null inputs
    // without crashing. Pair it with greedy to actually pick a token.
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let tokens = model.tokenize("hello", true, false);
    seq.extend(&tokens);

    let vocab = vocab_ptr_of(&model);
    let grammar = CString::new("root ::= \"a\" | \"b\"").unwrap();
    let root = CString::new("root").unwrap();

    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::grammar_lazy(
            vocab,
            grammar.as_ptr(),
            root.as_ptr(),
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            0,
        ))
        .add(Sampler::greedy());
    let token = seq.sample(&chain);
    assert!(
        token >= 0 && token < model.n_tokens(),
        "lazy-grammar sample should be in vocab range, got {token}"
    );
}
