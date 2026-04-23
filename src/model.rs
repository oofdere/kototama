use std::ops::{Deref, DerefMut};

use llama_sys::*;

use crate::ModelParams;

pub struct Model(*mut llama_model);

impl Model {
    pub fn load_from_file(path: &str, params: ModelParams) -> Self {
        // TODO: handle errors (null return, path not exist, etc.)
        let path = std::ffi::CString::new(path).unwrap();
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params.into()) };
        Self(model)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.0) };
    }
}

impl Deref for Model {
    type Target = *mut llama_sys::llama_model;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
 
impl DerefMut for Model {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_file() {
        let path = std::env::var("TEST_MODEL_PATH").unwrap_or_else(|_| "./model.gguf".to_string());
        let params = ModelParams::new();
        let model = Model::load_from_file(&path, params);
    }
}
