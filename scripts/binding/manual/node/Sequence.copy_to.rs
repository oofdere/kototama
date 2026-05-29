    /// Copy tokens in `[start, end)` from this sequence into `other`.
    #[napi(js_name = "copyTo")]
    pub fn copy_to(&self, other: &JsSequence, start: u32, end: u32) {
        // Guard against locking the same sequence twice (self-copy would deadlock).
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        let this = self.inner.lock().unwrap();
        let mut that = other.inner.lock().unwrap();
        this.copy_to(&mut *that, start as usize..end as usize);
    }
