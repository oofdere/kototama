use std::ops::{Deref, DerefMut};
use llama_sys::*;
use crate::Backend;

// eventually builder pattern
pub struct ModelParams(pub llama_sys::llama_model_params);


impl ModelParams {
    pub fn new() -> Self {
        Self(unsafe { llama_sys::llama_model_default_params() })
    }
}

impl Deref for ModelParams {
    type Target = llama_sys::llama_model_params;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
 
impl DerefMut for ModelParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<llama_sys::llama_model_params> for ModelParams {
    fn into(self) -> llama_sys::llama_model_params {
        self.0
    }
}

pub struct Model {
    model: *mut llama_model,
    pub vocab: *const llama_vocab,
    _backend: Backend,
}

impl Model {
    pub fn load_from_file(path: &str, params: ModelParams) -> Self {
        // TODO: handle errors (null return, path not exist, etc.)
        let _backend = Backend::acquire();
        let path = std::ffi::CString::new(path).unwrap();
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };
        let vocab = unsafe { llama_model_get_vocab(model) };
        Self { model, vocab, _backend }
    }

    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        let mut tokens = vec![0i32; text.len()]; // look into using smallvec/stack array
        let n_tokens = unsafe {
            llama_sys::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32, // is this needed it's a vec
                true,
                true,
            )
        };
        tokens.truncate(n_tokens as usize);
        tokens
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.model) };
    }
}

impl Deref for Model {
    type Target = *mut llama_sys::llama_model;
    fn deref(&self) -> &Self::Target {
        &self.model
    }
}
 
impl DerefMut for Model {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_params() {
        let _params = ModelParams::new();
    }

    #[test]
    fn load_from_file() {
        let path = std::env::var("TEST_MODEL_PATH").unwrap_or_else(|_| "./model.gguf".to_string());
        let params = ModelParams::new();
        let model = Model::load_from_file(&path, params);
        assert!(!model.vocab.is_null());
    }
}
