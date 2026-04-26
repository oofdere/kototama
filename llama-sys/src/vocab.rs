use crate::*;
use std::ffi::CStr;

macro_rules! token_option {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(&self) -> Option<llama_token> {
            let token = unsafe { $ffi_fn(self) };
            if token == LLAMA_TOKEN_NULL { None } else { Some(token) }
        }
    };
}

impl llama_vocab {
    token_option!(bos, llama_vocab_bos);
    token_option!(cls, llama_vocab_cls);
    token_option!(eos, llama_vocab_eos);
    token_option!(eot, llama_vocab_eot);
    token_option!(fim_mid, llama_vocab_fim_mid);
    token_option!(fim_pad, llama_vocab_fim_pad);
    token_option!(fim_pre, llama_vocab_fim_pre);
    token_option!(fim_rep, llama_vocab_fim_rep);
    token_option!(fim_sep, llama_vocab_fim_sep);
    token_option!(fim_suf, llama_vocab_fim_suf);

    pub fn get_add_bos(&self) -> bool {
        unsafe { llama_vocab_get_add_bos(self) }
    }

    pub fn get_add_eos(&self) -> bool {
        unsafe { llama_vocab_get_add_eos(self) }
    }

    pub fn get_add_sep(&self) -> bool {
        unsafe { llama_vocab_get_add_sep(self) }
    }

    pub fn get_attr(&self, token: llama_token) -> llama_token_attr {
        unsafe { llama_vocab_get_attr(self, token) }
    }

    pub fn get_score(&self, token: llama_token) -> f32 {
        unsafe { llama_vocab_get_score(self, token) }
    }

    pub fn get_text(&self, token: llama_token) -> &CStr {
        unsafe {
            let ptr = llama_vocab_get_text(self, token);
            CStr::from_ptr(ptr)
        }
    }

    pub fn is_control(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_control(self, token) }
    }

    pub fn is_eog(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_eog(self, token) }
    }

    token_option!(mask, llama_vocab_mask);

    pub fn n_tokens(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self) }
    }

    token_option!(nl, llama_vocab_nl);
    token_option!(pad, llama_vocab_pad);
    token_option!(sep, llama_vocab_sep);

    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self) }
    }
}