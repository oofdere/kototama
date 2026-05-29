    /// Copy KV cache cells in position range `[start, end)` into `other`.
    #[napi(js_name = "kvCopy")]
    pub fn kv_copy(&self, other: &JsSequence, start: i32, end: i32) {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        let this = self.inner.lock().unwrap();
        let mut that = other.inner.lock().unwrap();
        this.kv_copy(&mut *that, start..end);
    }
