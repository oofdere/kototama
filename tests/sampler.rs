mod common;

use rusty_llama::{Sampler, SamplerChain, SamplerChainParams};

// ---------- Sampler constructors ----------

#[test]
fn greedy_init() {
    let _s = Sampler::greedy();
}

#[test]
fn greedy_clone() {
    let s = Sampler::greedy();
    let _s2 = s.clone();
}

#[test]
fn dist_init() {
    let _s = Sampler::dist(42);
}

#[test]
fn temp_init() {
    let _s = Sampler::temp(0.8);
}

#[test]
fn top_k_init() {
    let _s = Sampler::top_k(40);
}

#[test]
fn top_p_init() {
    let _s = Sampler::top_p(0.95, 1);
}

#[test]
fn min_p_init() {
    let _s = Sampler::min_p(0.05, 1);
}

#[test]
fn mirostat_v2_init() {
    let _s = Sampler::mirostat_v2(42, 5.0, 0.1);
}

#[test]
fn penalties_init() {
    let _s = Sampler::penalties(64, 1.1, 0.0, 0.0);
}

#[test]
fn temp_ext_init() {
    let _s = Sampler::temp_ext(0.8, 0.1, 1.0);
}

#[test]
fn typical_init() {
    let _s = Sampler::typical(0.9, 1);
}

#[test]
fn xtc_init() {
    let _s = Sampler::xtc(0.1, 0.1, 1, 42);
}

#[test]
fn top_n_sigma_init() {
    let _s = Sampler::top_n_sigma(1.0);
}

#[test]
fn adaptive_p_init() {
    let _s = Sampler::adaptive_p(0.1, 0.9, 42);
}

// ---------- SamplerChain ----------

#[test]
fn sampler_chain_new() {
    let params = SamplerChainParams::new();
    let _chain = SamplerChain::new(&params);
}

#[test]
fn sampler_chain_add() {
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::top_k(40))
        .add(Sampler::top_p(0.95, 1))
        .add(Sampler::temp(0.8))
        .add(Sampler::dist(42));
    drop(chain);
}

#[test]
fn sampler_chain_perf() {
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());
    let _perf = chain.perf();
}

#[test]
fn sampler_chain_params_deref() {
    let params = SamplerChainParams::new();
    // Deref exposes the inner llama_sampler_chain_params — just check it's accessible
    let _no_perf = params.no_perf;
}

// ---------- Sampler::grammar* null-return guard ----------
//
// `llama_sampler_init_grammar*` returns `nullptr` when grammar parsing fails
// (parse error, missing root symbol, or left recursion). Before the guard,
// the wrapper would store that null pointer inside `Sampler`, and the
// `LlamaSampler::as_ptr() -> chain.add(...) -> seq.sample(&chain)` pipeline
// would feed the null sampler to `llama_sampler_sample` — UB inside
// llama.cpp. The constructors now return `Option<Self>` so the null
// cannot escape into safe code paths.

#[test]
fn grammar_returns_none_on_missing_root_symbol() {
    let model = common::load_model();
    // The vocab pointer is internal to the model; we only need it to satisfy
    // the FFI signature. Since the grammar is structurally invalid, the
    // failure happens before the vocab is consulted, so the model only needs
    // to be alive for the duration of the call (which it is — `model` is in
    // scope until end of function).
    let vocab = unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) };
    let grammar_str = std::ffi::CString::new("rule ::= [a-z]").unwrap();
    let root = std::ffi::CString::new("nonexistent").unwrap();
    let result = Sampler::grammar(vocab, grammar_str.as_ptr(), root.as_ptr());
    assert!(
        result.is_none(),
        "Sampler::grammar should return None when the grammar lacks the requested root symbol"
    );
}

#[test]
fn grammar_returns_some_for_valid_grammar() {
    let model = common::load_model();
    let vocab = unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) };
    let grammar_str = std::ffi::CString::new("root ::= [a-z]").unwrap();
    let root = std::ffi::CString::new("root").unwrap();
    let result = Sampler::grammar(vocab, grammar_str.as_ptr(), root.as_ptr());
    assert!(
        result.is_some(),
        "Sampler::grammar should return Some for a well-formed grammar"
    );
}

#[test]
fn grammar_lazy_returns_none_on_missing_root_symbol() {
    let model = common::load_model();
    let vocab = unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) };
    let grammar_str = std::ffi::CString::new("rule ::= [a-z]").unwrap();
    let root = std::ffi::CString::new("nonexistent").unwrap();
    let result = Sampler::grammar_lazy(
        vocab,
        grammar_str.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        0,
    );
    assert!(
        result.is_none(),
        "Sampler::grammar_lazy should return None when the grammar lacks the requested root symbol"
    );
}

#[test]
fn grammar_lazy_patterns_returns_none_on_missing_root_symbol() {
    let model = common::load_model();
    let vocab = unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) };
    let grammar_str = std::ffi::CString::new("rule ::= [a-z]").unwrap();
    let root = std::ffi::CString::new("nonexistent").unwrap();
    let result = Sampler::grammar_lazy_patterns(
        vocab,
        grammar_str.as_ptr(),
        root.as_ptr(),
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        0,
    );
    assert!(
        result.is_none(),
        "Sampler::grammar_lazy_patterns should return None when the grammar lacks the requested root symbol"
    );
}
