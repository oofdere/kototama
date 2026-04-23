use std::ops::{Deref, DerefMut};
use llama_sys::*;

use crate::Model;

pub struct ContextParams(llama_context_params);

// todo builder pattern
impl ContextParams {
    pub fn new() -> Self {
        Self(unsafe { llama_context_default_params() })
    }
}

pub struct Context<'a> {
    ctx: *mut llama_context,
    _model: std::marker::PhantomData<&'a Model>,
}

impl<'a> Context<'a> {
    pub fn new(model: &'a Model, params: ContextParams) -> Self {
        let ctx = unsafe { llama_init_from_model(**model, params.0) };
        Self { ctx, _model: std::marker::PhantomData }
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}

impl<'a> Deref for Context<'a> {
    type Target = *mut llama_sys::llama_context;
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}
 
impl<'a> DerefMut for Context<'a>  {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}