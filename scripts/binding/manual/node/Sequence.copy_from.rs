    /// Copy tokens in `[start, end)` from `other` into this sequence.
    #[napi(js_name = "copyFrom")]
    pub fn copy_from(&self, other: &JsSequence, start: u32, end: u32) {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        let mut this = self.inner.lock().unwrap();
        let that = other.inner.lock().unwrap();
        this.copy_from(&*that, start as usize..end as usize);
    }
