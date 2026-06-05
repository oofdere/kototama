// Based on alef-generated scaffolding, with manual fixes for:
// - Mutex<Sequence> (push/extend/pop/decode need &mut self)
// - i32 token types (alef extracted them as String due to bindgen aliases)
// - Real implementations for model_load_from_file, context_new, etc.
#![allow(dead_code, unused_imports, unused_variables)]
#![allow(clippy::too_many_arguments, clippy::let_unit_value, clippy::needless_borrow)]

use rustler::ResourceArc;
use rustler::Encoder;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Model {
    inner: Arc<rusty_llama::Model>,
}

impl std::panic::RefUnwindSafe for Model {}
impl rustler::Resource for Model {}

#[derive(Clone)]
pub struct Context {
    inner: Arc<rusty_llama::Context>,
}

impl std::panic::RefUnwindSafe for Context {}
impl rustler::Resource for Context {}

pub struct Sequence {
    inner: Mutex<rusty_llama::Sequence>,
}

impl std::panic::RefUnwindSafe for Sequence {}
impl rustler::Resource for Sequence {}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, rustler::NifUnitEnum)]
pub enum DecodeError {
    SlotNotFound,
    Aborted,
    InvalidInput,
    FatalError,
}

impl From<rusty_llama::DecodeError> for DecodeError {
    fn from(val: rusty_llama::DecodeError) -> Self {
        match val {
            rusty_llama::DecodeError::SlotNotFound => Self::SlotNotFound,
            rusty_llama::DecodeError::Aborted => Self::Aborted,
            rusty_llama::DecodeError::InvalidInput => Self::InvalidInput,
            rusty_llama::DecodeError::FatalError => Self::FatalError,
        }
    }
}

// -- Model --

#[rustler::nif]
pub fn model_load_from_file(path: String, n_gpu_layers: i32) -> Result<ResourceArc<Model>, String> {
    let mut params = rusty_llama::ModelParams::new();
    params.set_n_gpu_layers(n_gpu_layers);
    let model = rusty_llama::Model::load_from_file(&path, params)
        .map_err(|_| "failed to load model".to_string())?;
    Ok(ResourceArc::new(Model { inner: Arc::new(model) }))
}

#[rustler::nif]
pub fn model_chat_template(resource: ResourceArc<Model>, name: Option<String>) -> Option<String> {
    resource.inner.chat_template(name.as_deref())
}

#[rustler::nif]
pub fn model_desc(resource: ResourceArc<Model>) -> String {
    resource.inner.desc()
}

#[rustler::nif]
pub fn model_has_decoder(resource: ResourceArc<Model>) -> bool {
    resource.inner.has_decoder()
}

#[rustler::nif]
pub fn model_decoder_start_token(resource: ResourceArc<Model>) -> Option<i32> {
    resource.inner.decoder_start_token()
}

#[rustler::nif]
pub fn model_has_encoder(resource: ResourceArc<Model>) -> bool {
    resource.inner.has_encoder()
}

#[rustler::nif]
pub fn model_is_diffusion(resource: ResourceArc<Model>) -> bool {
    resource.inner.is_diffusion()
}

#[rustler::nif]
pub fn model_is_hybrid(resource: ResourceArc<Model>) -> bool {
    resource.inner.is_hybrid()
}

#[rustler::nif]
pub fn model_is_recurrent(resource: ResourceArc<Model>) -> bool {
    resource.inner.is_recurrent()
}

#[rustler::nif]
pub fn model_token_to_piece(resource: ResourceArc<Model>, token: i32) -> Result<String, String> {
    resource.inner.token_to_piece(token)
        .map_err(|_| "token_to_piece failed".to_string())
}

#[rustler::nif]
pub fn model_tokenize(resource: ResourceArc<Model>, text: String, add_special: bool, parse_special: bool) -> Vec<i32> {
    resource.inner.tokenize(&text, add_special, parse_special)
}

#[rustler::nif]
pub fn model_get_add_bos(resource: ResourceArc<Model>) -> bool {
    resource.inner.get_add_bos()
}

#[rustler::nif]
pub fn model_get_add_eos(resource: ResourceArc<Model>) -> bool {
    resource.inner.get_add_eos()
}

#[rustler::nif]
pub fn model_get_add_sep(resource: ResourceArc<Model>) -> bool {
    resource.inner.get_add_sep()
}

#[rustler::nif]
pub fn model_get_score(resource: ResourceArc<Model>, token: i32) -> f32 {
    resource.inner.get_score(token)
}

#[rustler::nif]
pub fn model_get_text(resource: ResourceArc<Model>, token: i32) -> String {
    resource.inner.get_text(token).to_string_lossy().into_owned()
}

#[rustler::nif]
pub fn model_is_control(resource: ResourceArc<Model>, token: i32) -> bool {
    resource.inner.is_control(token)
}

#[rustler::nif]
pub fn model_is_eog(resource: ResourceArc<Model>, token: i32) -> bool {
    resource.inner.is_eog(token)
}

#[rustler::nif]
pub fn model_n_tokens(resource: ResourceArc<Model>) -> i32 {
    resource.inner.n_tokens()
}

// -- Context --

#[rustler::nif]
pub fn context_new(model: ResourceArc<Model>, n_ctx: u32, n_batch: u32) -> Result<ResourceArc<Context>, String> {
    let mut params = rusty_llama::ContextParams::new();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    let ctx = rusty_llama::Context::new(&model.inner, &params)
        .map_err(|_| "failed to create context".to_string())?;
    Ok(ResourceArc::new(Context { inner: Arc::new(ctx) }))
}

#[rustler::nif]
pub fn context_sequence(resource: ResourceArc<Context>) -> Option<ResourceArc<Sequence>> {
    resource.inner.sequence().map(|v| ResourceArc::new(Sequence { inner: Mutex::new(v) }))
}

#[rustler::nif]
pub fn context_free_slots(resource: ResourceArc<Context>) -> usize {
    resource.inner.free_slots()
}

#[rustler::nif]
pub fn context_n_ctx(resource: ResourceArc<Context>) -> u32 {
    resource.inner.n_ctx()
}

#[rustler::nif]
pub fn context_can_shift(resource: ResourceArc<Context>) -> bool {
    resource.inner.can_shift()
}

// -- Sequence --

#[rustler::nif]
pub fn sequence_logits(resource: ResourceArc<Sequence>) -> Option<Vec<f32>> {
    resource.inner.lock().unwrap().logits().map(|s| s.to_vec())
}

#[rustler::nif]
pub fn sequence_is_empty(resource: ResourceArc<Sequence>) -> bool {
    resource.inner.lock().unwrap().is_empty()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn sequence_push(resource: ResourceArc<Sequence>, token: i32) {
    resource.inner.lock().unwrap().push(token);
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn sequence_decode(resource: ResourceArc<Sequence>) {
    resource.inner.lock().unwrap().decode();
}

#[rustler::nif]
pub fn sequence_pop(resource: ResourceArc<Sequence>) -> Option<i32> {
    resource.inner.lock().unwrap().pop()
}

#[rustler::nif]
pub fn sequence_len(resource: ResourceArc<Sequence>) -> usize {
    resource.inner.lock().unwrap().len()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn sequence_extend(resource: ResourceArc<Sequence>, tokens: Vec<i32>) {
    resource.inner.lock().unwrap().extend(&tokens);
}

#[rustler::nif]
pub fn sequence_get(resource: ResourceArc<Sequence>, index: usize) -> Option<i32> {
    resource.inner.lock().unwrap().get(index)
}

#[rustler::nif]
pub fn sequence_pos_min(resource: ResourceArc<Sequence>) -> i32 {
    resource.inner.lock().unwrap().pos_min()
}

#[rustler::nif]
pub fn sequence_pos_max(resource: ResourceArc<Sequence>) -> i32 {
    resource.inner.lock().unwrap().pos_max()
}

#[rustler::nif]
pub fn sequence_tokens(resource: ResourceArc<Sequence>) -> Vec<i32> {
    resource.inner.lock().unwrap().tokens().to_vec()
}

fn on_load(env: rustler::Env, _info: rustler::Term) -> bool {
    env.register::<Model>().expect("Failed to register resource type Model");
    env.register::<Context>().expect("Failed to register resource type Context");
    env.register::<Sequence>().expect("Failed to register resource type Sequence");
    true
}

rustler::init!("Elixir.RustyLlama.Native", load = on_load);
