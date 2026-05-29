    #[napi(factory)]
    pub fn new(model: &JsModel, n_ctx: u32, n_batch: u32) -> Result<JsContext> {
        let mut params = rusty_llama::ContextParams::new();
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        let ctx = rusty_llama::Context::new(&model.inner, &params)
            .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "failed to create context"))?;
        Ok(JsContext { inner: Arc::new(ctx) })
    }
