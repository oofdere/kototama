use crate::*;
use llama_sys::*;
use std::ffi::CStr;

macro_rules! token_option {
    ($name:ident, $ffi_fn:ident) => {
        #[inline]
        pub fn $name(&self) -> Option<llama_token> {
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

    #[inline]
    pub fn get_attr(&self, token: llama_token) -> llama_token_attr {
        unsafe { llama_vocab_get_attr(self.vocab_ptr(), token) }
    }

    #[inline]
    pub fn get_score(&self, token: llama_token) -> f32 {
        unsafe { llama_vocab_get_score(self.vocab_ptr(), token) }
    }

    #[inline]
    pub fn get_text(&self, token: llama_token) -> &CStr {
        unsafe {
            let ptr = llama_vocab_get_text(self.vocab_ptr(), token);
            CStr::from_ptr(ptr)
        }
    }

    #[inline]
    pub fn is_control(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_control(self.vocab_ptr(), token) }
    }

    #[inline]
    pub fn is_eog(&self, token: llama_token) -> bool {
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
