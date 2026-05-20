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

#[test]
fn sampler_chain_params_deref_mut() {
    let mut params = SamplerChainParams::new();
    (*params).no_perf = true;
    assert!(params.no_perf);
}

#[test]
fn sampler_chain_params_as_ptr_not_null() {
    let params = SamplerChainParams::new();
    assert!(!params.as_ptr().is_null());
}

#[test]
fn sampler_chain_params_as_mut_ptr_not_null() {
    let mut params = SamplerChainParams::new();
    assert!(!params.as_mut_ptr().is_null());
}

// ---------- Sampler constructors that need a vocab pointer ----------

#[test]
fn mirostat_init() {
    let model = common::load_model();
    let _s = Sampler::mirostat(model.n_tokens(), 42, 5.0, 0.1, 100);
}

#[test]
fn logit_bias_empty_init() {
    let model = common::load_model();
    // With n_logit_bias = 0 the pointer is unused, so a null pointer is fine.
    let _s = Sampler::logit_bias(model.n_tokens(), 0, std::ptr::null());
}

#[test]
fn infill_init() {
    let model = common::load_model();
    let _s = Sampler::infill(model.vocab);
}

#[test]
fn dry_init_no_breakers() {
    let model = common::load_model();
    // num_breakers = 0 means seq_breakers is unused; a null pointer is acceptable.
    let _s = Sampler::dry(
        model.vocab,
        /* n_ctx_train */ 2048,
        /* dry_multiplier */ 0.0,
        /* dry_base */ 1.75,
        /* dry_allowed_length */ 2,
        /* dry_penalty_last_n */ 64,
        std::ptr::null_mut(),
        0,
    );
}
