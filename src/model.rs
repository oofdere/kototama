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
    pub fn load_from_file(path: &str, params: ModelParams) -> Result<Self, ()> {
        // TODO: handle errors (null return, path not exist, etc.)
        let _backend = Backend::acquire();
        let path = std::ffi::CString::new(path).map_err(|_| ())?;
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };

        if model.is_null() {
            return Err(());
        }

        let vocab = unsafe { llama_model_get_vocab(model) };

        Ok(Self {
            model,
            vocab,
            _backend,
        })
    }

    /// gets the chat template of the specified name, or the default if None
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let name_cstr = name.map(|s| std::ffi::CString::new(s).unwrap());
        let name_ptr = name_cstr.as_ref().map(|s| s.as_ptr()).unwrap_or(null());
        let str = unsafe { llama_model_chat_template(self.model, name_ptr) };
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
    #[inline]
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.model) }
    }

    /// For encoder-decoder models, this function returns id of the token that must be provided to the decoder to start generating output sequence. For other models, it returns -1.
    #[inline]
    pub fn decoder_start_token(&self) -> Option<llama_token> {
        let token = unsafe { llama_model_decoder_start_token(self.model) };
        if token == LLAMA_TOKEN_NULL {
            None
        } else {
            Some(token)
        }
    }

    /// Returns true if the model contains an encoder that requires llama_encode() call
    #[inline]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.model) }
    }

    /// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    #[inline]
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.model) }
    }

    /// Returns true if the model is hybrid (like Jamba, Granite, etc.)
    #[inline]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.model) }
    }

    /// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.model) }
    }

    /// Convert a token to its text representation
    pub fn token_to_piece(&self, token: i32) -> Result<String, ()> {
        let mut buf = [0u8; 64]; // look into setting this dynamically from the model's vocab
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
        if n < 0 {
            return Err(());
        }
        Ok(String::from_utf8_lossy(&buf[..n as usize]).to_string())
    }

    /// tokenize text
    /// @details Convert the provided text into tokens.
    ///
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<llama_token> {
        let len = -unsafe {
            llama_sys::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        let mut tokens = vec![0i32; len as usize]; // look into using smallvec/stack array
        let n_tokens = unsafe {
            llama_sys::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32, // is this needed it's a vec
                add_special,
                parse_special,
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
        let model = Model::load_from_file(&path, params).unwrap();
        assert!(!model.vocab.is_null());
    }
}
