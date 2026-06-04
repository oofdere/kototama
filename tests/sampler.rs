use rusty_llama::{LlamaSampler, Sampler, SamplerChain, SamplerChainParams};

mod common;

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

// ---------- unsafe-fn constructors taking raw FFI pointers ----------
//
// `Sampler::infill`, `logit_bias`, `dry`, `grammar`, `grammar_lazy`, and
// `grammar_lazy_patterns` accept raw pointers (`*const llama_vocab`,
// `*const c_char`, …) that `llama.cpp` dereferences during initialization.
// They are `unsafe fn` so that callers must affirm — in an `unsafe` block —
// that those pointers are non-null, point at valid data, and outlive the
// call. If anyone relaxes them back to safe `fn`, this test stops needing
// the `unsafe` block and the `#![deny(unused_unsafe)]` at the top of the
// test would catch the regression.

#[deny(unused_unsafe)]
#[test]
fn grammar_constructor_is_unsafe_fn() {
    let model = common::load_model();
    let vocab = unsafe { llama_sys::llama_model_get_vocab(model.as_ptr()) };

    // A trivially-satisfiable grammar: any single character.
    let grammar = std::ffi::CString::new("root ::= [a-z]").unwrap();
    let root = std::ffi::CString::new("root").unwrap();

    // SAFETY: `vocab` belongs to `model`, which stays alive for the body
    // of this scope; `grammar` and `root` are valid NUL-terminated CStrings
    // held on the stack across the call.
    let sampler =
        unsafe { Sampler::grammar(vocab, grammar.as_ptr(), root.as_ptr()) };
    assert!(!sampler.as_ptr().is_null());
}
