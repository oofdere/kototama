//! `llama_token_attr` is a bitmask on the C side: llama.cpp routinely ORs
//! flags together (e.g. a control-looking token that was typed NORMAL gets
//! `NORMAL | CONTROL`). These tests exercise `Model::get_attr` with such
//! combined values, which are unrepresentable in a Rust `enum` binding.

mod common;

use llama_sys::llama_token_attr;
use rusty_llama::{Model, ModelParams};

/// Build a copy of the test model whose token 166 (`"start"`, a NORMAL token)
/// is renamed to `"<eos>"`. llama.cpp's vocab loader detects the EOG-looking
/// name and ORs `LLAMA_TOKEN_ATTR_CONTROL` onto the existing NORMAL attribute,
/// so `get_attr(166)` returns the combined bitmask `NORMAL | CONTROL`.
fn patched_model() -> Model {
    let bytes = std::fs::read(common::model_path()).expect("test model missing");
    // GGUF string: u64 little-endian length prefix followed by the bytes.
    let needle = b"\x05\x00\x00\x00\x00\x00\x00\x00start";
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("expected token not found in test model");
    let mut bytes = bytes;
    bytes[pos + 8..pos + 13].copy_from_slice(b"<eos>");

    let path = std::env::temp_dir().join("rusty-llama-token-attr-test.gguf");
    std::fs::write(&path, &bytes).expect("failed to write patched model");

    let mut params = ModelParams::new();
    params.n_gpu_layers = 0;
    Model::load_from_file(path.to_str().unwrap(), params).expect("failed to load patched model")
}

const NORMAL: llama_token_attr = llama_token_attr::LLAMA_TOKEN_ATTR_NORMAL;
const CONTROL: llama_token_attr = llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL;

#[test]
fn get_attr_returns_combined_flags() {
    let model = patched_model();
    let attr = model.get_attr(166);
    assert_eq!(attr, NORMAL | CONTROL);
}

#[test]
fn combined_flags_can_be_tested_individually() {
    let model = patched_model();
    let attr = model.get_attr(166);
    assert_ne!(attr & NORMAL, llama_token_attr(0));
    assert_ne!(attr & CONTROL, llama_token_attr(0));
    assert_eq!(
        attr & llama_token_attr::LLAMA_TOKEN_ATTR_BYTE,
        llama_token_attr(0)
    );
}

#[test]
fn get_attr_single_flags_unchanged() {
    let model = common::load_model();
    // <unk>/<s>/</s> are CONTROL tokens; ordinary vocab entries are NORMAL.
    assert_eq!(model.get_attr(0), CONTROL);
    assert_eq!(model.get_attr(166), NORMAL);
}
