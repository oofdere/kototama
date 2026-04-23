use std::ops::{Deref, DerefMut};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init() {
        let _params = ModelParams::new();
    }
}