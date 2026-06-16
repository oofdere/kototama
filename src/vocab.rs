use crate::*;
use llama_sys::*;
use std::ffi::CStr;

macro_rules! token_option {
    ($name:ident, $ffi_fn:ident) => {
        #[inline]
        pub fn $name(&self) -> Option<i32> {
            let token = unsafe { $ffi_fn(self.vocab_ptr()) };
            if token == LLAMA_TOKEN_NULL {
                None
            } else {
                Some(token)
            }
        }
    };
}

impl Model {
    token_option!(bos_token, llama_vocab_bos);
    token_option!(cls_token, llama_vocab_cls);
    token_option!(eos_token, llama_vocab_eos);
    token_option!(eot_token, llama_vocab_eot);
    token_option!(fim_mid_token, llama_vocab_fim_mid);
    token_option!(fim_pad_token, llama_vocab_fim_pad);
    token_option!(fim_pre_token, llama_vocab_fim_pre);
    token_option!(fim_rep_token, llama_vocab_fim_rep);
    token_option!(fim_sep_token, llama_vocab_fim_sep);
    token_option!(fim_suf_token, llama_vocab_fim_suf);

    #[inline]
    pub fn get_add_bos(&self) -> bool {
        unsafe { llama_vocab_get_add_bos(self.vocab_ptr()) }
    }

    #[inline]
    pub fn get_add_eos(&self) -> bool {
        unsafe { llama_vocab_get_add_eos(self.vocab_ptr()) }
    }

    #[inline]
    pub fn get_add_sep(&self) -> bool {
        unsafe { llama_vocab_get_add_sep(self.vocab_ptr()) }
    }

    /// True iff `token` is a valid index into this vocab (`0 <= token < n_tokens()`).
    #[inline]
    pub(crate) fn token_in_range(&self, token: i32) -> bool {
        token >= 0 && token < self.n_tokens()
    }

    #[inline]
    pub fn get_attr(&self, token: i32) -> Option<llama_token_attr> {
        if !self.token_in_range(token) {
            return None;
        }
        Some(unsafe { llama_vocab_get_attr(self.vocab_ptr(), token) })
    }

    #[inline]
    pub fn get_score(&self, token: i32) -> Option<f32> {
        if !self.token_in_range(token) {
            return None;
        }
        Some(unsafe { llama_vocab_get_score(self.vocab_ptr(), token) })
    }

    #[inline]
    pub fn get_text(&self, token: i32) -> Option<&CStr> {
        if !self.token_in_range(token) {
            return None;
        }
        let ptr = unsafe { llama_vocab_get_text(self.vocab_ptr(), token) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) })
    }

    #[inline]
    pub fn is_control(&self, token: i32) -> bool {
        if !self.token_in_range(token) {
            return false;
        }
        unsafe { llama_vocab_is_control(self.vocab_ptr(), token) }
    }

    /// Safe for any `i32`. llama.cpp's `is_eog` is a `LLAMA_TOKEN_NULL` check
    /// plus a set-membership lookup — it does not index `id_to_token`.
    #[inline]
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab_ptr(), token) }
    }

    token_option!(mask_token, llama_vocab_mask);

    #[inline]
    pub fn n_tokens(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self.vocab_ptr()) }
    }

    token_option!(nl_token, llama_vocab_nl);
    token_option!(pad_token, llama_vocab_pad);
    token_option!(sep_token, llama_vocab_sep);

    #[inline]
    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self.vocab_ptr()) }
    }
}
