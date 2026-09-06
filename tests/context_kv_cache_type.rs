//! `ContextParams::type_k` / `type_v` accept any `ggml_type`, but ggml only
//! implements KV caches for float and quantized types. Integer and 64-bit
//! float caches used to pass `Context::new` and then abort/segfault inside
//! ggml on the first `Sequence::push`. `Context::new` must reject them.

use llama_sys::ggml_type;
use rusty_llama::test_common::load_model_and_context;
use rusty_llama::Context;

const UNSUPPORTED: &[ggml_type] = &[
    ggml_type::GGML_TYPE_I8,
    ggml_type::GGML_TYPE_I16,
    ggml_type::GGML_TYPE_I32,
    ggml_type::GGML_TYPE_I64,
    ggml_type::GGML_TYPE_F64,
    ggml_type::GGML_TYPE_COUNT,
];

#[test]
fn unsupported_type_k_is_rejected() {
    let (model, mut params) = load_model_and_context();
    for &t in UNSUPPORTED {
        params.type_k = t;
        params.type_v = ggml_type::GGML_TYPE_F16;
        assert!(Context::new(&model, &params).is_err(), "type_k = {t:?}");
    }
}

#[test]
fn unsupported_type_v_is_rejected() {
    let (model, mut params) = load_model_and_context();
    for &t in UNSUPPORTED {
        params.type_k = ggml_type::GGML_TYPE_F16;
        params.type_v = t;
        assert!(Context::new(&model, &params).is_err(), "type_v = {t:?}");
    }
}

#[test]
fn float_types_still_decode() {
    let (model, mut params) = load_model_and_context();
    for t in [
        ggml_type::GGML_TYPE_F32,
        ggml_type::GGML_TYPE_F16,
        ggml_type::GGML_TYPE_BF16,
    ] {
        params.type_k = t;
        params.type_v = t;
        let ctx = Context::new(&model, &params).unwrap_or_else(|_| panic!("{t:?}"));
        let mut seq = ctx.sequence().unwrap();
        seq.push(1);
        seq.push(2);
        assert_eq!(seq.logits().unwrap().len(), model.n_tokens() as usize);
    }
}

#[test]
fn quantized_types_are_left_to_llama_cpp() {
    // llama.cpp validates quantized cache types itself (block size vs head
    // size, flash attention requirements); we must not abort either way.
    let (model, mut params) = load_model_and_context();
    params.type_k = ggml_type::GGML_TYPE_Q8_0;
    params.type_v = ggml_type::GGML_TYPE_F16;
    let _ = Context::new(&model, &params);
}
