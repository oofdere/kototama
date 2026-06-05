// Based on alef-generated scaffolding, with manual fixes for:
// - i32 token types (alef extracted them as String due to bindgen aliases)
// - Real implementations for load_from_file, context_new, etc.
// - Correct logits return type (f64[] for JS)
#![allow(dead_code, unused_imports, unused_variables)]
#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::let_unit_value, clippy::needless_borrow)]

use napi::*;
use napi_derive::napi;
use std::sync::Arc;
use std::sync::Mutex;

/// Thread-safe handle to a loaded model.
#[derive(Clone)]
#[napi(js_name = "Model")]
pub struct JsModel {
    inner: Arc<rusty_llama::Model>,
}

#[napi]
impl JsModel {
    #[napi(factory, js_name = "loadFromFile")]
    pub fn load_from_file(path: String, n_gpu_layers: i32) -> Result<JsModel> {
        let mut params = rusty_llama::ModelParams::new();
        params.set_n_gpu_layers(n_gpu_layers);
        let model = rusty_llama::Model::load_from_file(&path, params)
            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "failed to load model"))?;
        Ok(JsModel { inner: Arc::new(model) })
    }

    #[napi(js_name = "chatTemplate")]
    pub fn chat_template(&self, name: Option<String>) -> Option<String> {
        self.inner.chat_template(name.as_deref())
    }

    #[napi]
    pub fn desc(&self) -> String {
        self.inner.desc()
    }

    #[napi(js_name = "hasDecoder")]
    pub fn has_decoder(&self) -> bool {
        self.inner.has_decoder()
    }

    #[napi(js_name = "decoderStartToken")]
    pub fn decoder_start_token(&self) -> Option<i32> {
        self.inner.decoder_start_token()
    }

    #[napi(js_name = "hasEncoder")]
    pub fn has_encoder(&self) -> bool {
        self.inner.has_encoder()
    }

    #[napi(js_name = "isDiffusion")]
    pub fn is_diffusion(&self) -> bool {
        self.inner.is_diffusion()
    }

    #[napi(js_name = "isHybrid")]
    pub fn is_hybrid(&self) -> bool {
        self.inner.is_hybrid()
    }

    #[napi(js_name = "isRecurrent")]
    pub fn is_recurrent(&self) -> bool {
        self.inner.is_recurrent()
    }

    #[napi(js_name = "tokenToPiece")]
    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        self.inner.token_to_piece(token)
            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "token_to_piece failed"))
    }

    #[napi]
    pub fn tokenize(&self, text: String, add_special: bool, parse_special: bool) -> Vec<i32> {
        self.inner.tokenize(&text, add_special, parse_special)
    }

    #[napi(js_name = "getAddBos")]
    pub fn get_add_bos(&self) -> bool {
        self.inner.get_add_bos()
    }

    #[napi(js_name = "getAddEos")]
    pub fn get_add_eos(&self) -> bool {
        self.inner.get_add_eos()
    }

    #[napi(js_name = "getAddSep")]
    pub fn get_add_sep(&self) -> bool {
        self.inner.get_add_sep()
    }

    #[napi(js_name = "getScore")]
    pub fn get_score(&self, token: i32) -> f64 {
        self.inner.get_score(token) as f64
    }

    #[napi(js_name = "getText")]
    pub fn get_text(&self, token: i32) -> String {
        self.inner.get_text(token).to_string_lossy().into_owned()
    }

    #[napi(js_name = "isControl")]
    pub fn is_control(&self, token: i32) -> bool {
        self.inner.is_control(token)
    }

    #[napi(js_name = "isEog")]
    pub fn is_eog(&self, token: i32) -> bool {
        self.inner.is_eog(token)
    }

    #[napi(js_name = "nTokens")]
    pub fn n_tokens(&self) -> i32 {
        self.inner.n_tokens()
    }
}

/// Handle to a running context actor.
#[derive(Clone)]
#[napi(js_name = "Context")]
pub struct JsContext {
    inner: Arc<rusty_llama::Context>,
}

#[napi]
impl JsContext {
    #[napi(factory)]
    pub fn new(model: &JsModel, n_ctx: u32, n_batch: u32) -> Result<JsContext> {
        let mut params = rusty_llama::ContextParams::new();
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        let ctx = rusty_llama::Context::new(&model.inner, &params)
            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "failed to create context"))?;
        Ok(JsContext { inner: Arc::new(ctx) })
    }

    #[napi]
    pub fn sequence(&self) -> Option<JsSequence> {
        self.inner.sequence().map(|v| JsSequence { inner: Arc::new(Mutex::new(v)) })
    }

    #[napi(js_name = "freeSlots")]
    pub fn free_slots(&self) -> i64 {
        self.inner.free_slots() as i64
    }

    #[napi(js_name = "nCtx")]
    pub fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    #[napi(js_name = "canShift")]
    pub fn can_shift(&self) -> bool {
        self.inner.can_shift()
    }
}

/// A sequence handle.
#[derive(Clone)]
#[napi(js_name = "Sequence")]
pub struct JsSequence {
    inner: Arc<Mutex<rusty_llama::Sequence>>,
}

#[napi]
impl JsSequence {
    #[napi]
    pub fn logits(&self) -> Option<Vec<f64>> {
        self.inner.lock().unwrap().logits().map(|s| s.iter().map(|&v| v as f64).collect())
    }

    #[napi(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    #[napi]
    pub fn push(&self, token: i32) {
        self.inner.lock().unwrap().push(token);
    }

    #[napi]
    pub fn decode(&self) {
        self.inner.lock().unwrap().decode();
    }

    #[napi]
    pub fn pop(&self) -> Option<i32> {
        self.inner.lock().unwrap().pop()
    }

    #[napi]
    pub fn len(&self) -> i64 {
        self.inner.lock().unwrap().len() as i64
    }

    #[napi]
    pub fn extend(&self, tokens: Vec<i32>) {
        self.inner.lock().unwrap().extend(&tokens);
    }

    #[napi]
    pub fn get(&self, index: u32) -> Option<i32> {
        self.inner.lock().unwrap().get(index as usize)
    }

    #[napi(js_name = "posMin")]
    pub fn pos_min(&self) -> i32 {
        self.inner.lock().unwrap().pos_min()
    }

    #[napi(js_name = "posMax")]
    pub fn pos_max(&self) -> i32 {
        self.inner.lock().unwrap().pos_max()
    }

    #[napi]
    pub fn tokens(&self) -> Vec<i32> {
        self.inner.lock().unwrap().tokens().to_vec()
    }
}

#[napi(string_enum, js_name = "DecodeError")]
#[derive(Clone)]
pub enum JsDecodeError {
    SlotNotFound,
    Aborted,
    InvalidInput,
    FatalError,
}
