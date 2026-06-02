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
fn sampler_chain_into_raw_does_not_double_free() {
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::greedy());
    let ptr = chain.into_raw();
    assert!(!ptr.is_null());
    // Manually free — if into_raw didn't prevent the destructor, the pointer
    // would already be freed and this would be a double-free / use-after-free.
    unsafe { llama_sys::llama_sampler_free(ptr) };
}

#[test]
fn sampler_chain_into_raw_with_multiple_samplers() {
    let chain = SamplerChain::new(&SamplerChainParams::new())
        .add(Sampler::top_k(40))
        .add(Sampler::top_p(0.95, 1))
        .add(Sampler::temp(0.8))
        .add(Sampler::dist(42));
    let ptr = chain.into_raw();
    assert!(!ptr.is_null());
    unsafe { llama_sys::llama_sampler_free(ptr) };
}
