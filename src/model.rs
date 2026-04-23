use crate::Backend;
use llama_sys::*;
use std::{
    ffi::c_char,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

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
        Self {
            model,
            vocab,
            _backend,
        }
    }

    /// gets the chat template of the specified name, or the default if None
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let str = unsafe {
            llama_model_chat_template(
                self.model,
                name.map(|s| s.as_ptr() as *const c_char).unwrap_or(null()),
            )
        };
        if str.is_null() {
            None
        } else {
            // convert to string
            let cstr = unsafe { std::ffi::CStr::from_ptr(str) };
            Some(cstr.to_string_lossy().to_string())
        }
    }

    /// get the model description
    pub fn desc(&self) -> String {
        // get length of name
        let len = unsafe { llama_model_desc(self.model, null_mut(), 0) };

        let mut buf = vec![c_char::default(); len as usize];
        unsafe { llama_model_desc(self.model, buf.as_mut_ptr(), len.try_into().unwrap()) };
        // throw it in a string
        let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        cstr.to_string_lossy().to_string()
    }

    /// Returns true if the model contains a decoder that requires llama_decode() call
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.model) }
    }

    /// For encoder-decoder models, this function returns id of the token that must be provided to the decoder to start generating output sequence. For other models, it returns -1.
    pub fn decoder_start_token(&self) -> i32 {
        unsafe { llama_model_decoder_start_token(self.model) }
    }

    /// Returns true if the model contains an encoder that requires llama_encode() call
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.model) }
    }

    /// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.model) }
    }

    /// Returns true if the model is hybrid (like Jamba, Granite, etc.)
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.model) }
    }

    /// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.model) }
    }

    /// Returns the beginning‑of‑sentence token for a given vocabulary.
    pub fn bos_token(&self) -> i32 {
        unsafe { llama_vocab_bos(self.vocab) }
    }

    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab, token) }
    }

    /// Convert a token to its text representation
    pub fn token_to_piece(&self, token: i32) -> String {
        let mut buf = [0u8; 64];
        let n = unsafe {
            llama_sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                true,
            )
        };
        String::from_utf8_lossy(&buf[..n as usize]).to_string()
    }

    /// tokenize text
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        // TODO: handle cases where the buffer is too small
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
