    #[napi(factory, js_name = "loadFromFile")]
    pub fn load_from_file(path: String, n_gpu_layers: i32) -> Result<JsModel> {
        let mut params = rusty_llama::ModelParams::new();
        params.n_gpu_layers = n_gpu_layers;
        let model = rusty_llama::Model::load_from_file(&path, params)
            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "failed to load model"))?;
        Ok(JsModel { inner: Arc::new(model) })
    }
