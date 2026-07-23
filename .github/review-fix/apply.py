from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:80]!r}")
    file.write_text(text.replace(old, new, 1))


replace_once(
    "src/context.rs",
    """pub(crate) struct SequenceSnapshot {
    pub tokens: Arc<[Token]>,
    pub logits: Option<Arc<[f32]>>,
}

impl SequenceSnapshot {
    fn empty() -> Self {
        Self {
            tokens: Arc::from([]),
            logits: None,
        }
    }
}
""",
    """pub(crate) struct SequenceSnapshot {
    pub tokens: Vec<Token>,
    pub token_snapshot: Option<Arc<[Token]>>,
    pub logits: Option<Arc<[f32]>>,
}

impl SequenceSnapshot {
    fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            token_snapshot: Some(Arc::from([])),
            logits: None,
        }
    }
}
""",
)

replace_once(
    "src/context.rs",
    """        Some(Arc::from(
            unsafe { std::slice::from_raw_parts(ptr, self.n_vocab as usize) }.to_vec(),
        ))
""",
    """        Some(Arc::from(unsafe {
            std::slice::from_raw_parts(ptr, self.n_vocab as usize)
        }))
""",
)

replace_once(
    "src/context.rs",
    """                    if let Ok(logits) = &result {
                        let mut state = snapshot.write().unwrap();
                        let mut tokens = state.tokens.to_vec();
                        tokens.push(token);
                        state.tokens = Arc::from(tokens);
                        state.logits = Some(logits.clone());
                    }
""",
    """                    if let Ok(logits) = &result {
                        let mut state = snapshot.write().unwrap();
                        state.tokens.push(token);
                        state.token_snapshot = None;
                        state.logits = Some(logits.clone());
                    }
""",
)

replace_once(
    "src/context.rs",
    """                        if ok {
                            let mut state = snapshot.write().unwrap();
                            let mut tokens = state.tokens.to_vec();
                            let token = tokens.pop();
                            state.tokens = Arc::from(tokens);
                            state.logits = None;
                            token
                        } else {
""",
    """                        if ok {
                            let mut state = snapshot.write().unwrap();
                            let token = state.tokens.pop();
                            state.token_snapshot = None;
                            state.logits = None;
                            token
                        } else {
""",
)

replace_once(
    "src/context.rs",
    """                        let mut state = snapshot.write().unwrap();
                        let mut tokens = state.tokens.to_vec();
                        tokens.drain(start as usize..end as usize);
                        state.tokens = Arc::from(tokens);
                        state.logits = None;
""",
    """                        let mut state = snapshot.write().unwrap();
                        state.tokens.drain(start as usize..end as usize);
                        state.token_snapshot = None;
                        state.logits = None;
""",
)

replace_once(
    "src/context.rs",
    """                        let mut dst_state = dst_snapshot.write().unwrap();
                        dst_state.tokens =
                            Arc::from(src_tokens[start as usize..end as usize].to_vec());
                        dst_state.logits = None;
""",
    """                        let mut dst_state = dst_snapshot.write().unwrap();
                        dst_state.tokens = src_tokens[start as usize..end as usize].to_vec();
                        dst_state.token_snapshot = None;
                        dst_state.logits = None;
""",
)

replace_once(
    "src/sequence.rs",
    """    pub(crate) fn new(ctx: Context, reservation: SequenceReservation) -> Self {
        let (id, snapshot) = reservation.into_parts();
        Self { ctx, id, snapshot }
    }

""",
    """    pub(crate) fn new(ctx: Context, reservation: SequenceReservation) -> Self {
        let (id, snapshot) = reservation.into_parts();
        Self { ctx, id, snapshot }
    }

    fn checked_range(range: Range<usize>) -> Option<Range<i32>> {
        let start = i32::try_from(range.start).ok()?;
        let end = i32::try_from(range.end).ok()?;
        Some(start..end)
    }

""",
)

replace_once(
    "src/sequence.rs",
    """    pub fn remove(&mut self, range: Range<usize>) -> bool {
        self.ctx.remove(
            self.id,
            range.start as i32,
            range.end as i32,
            self.snapshot.clone(),
        )
    }

    pub async fn remove_async(&mut self, range: Range<usize>) -> bool {
        self.ctx
            .remove_async(
                self.id,
                range.start as i32,
                range.end as i32,
                self.snapshot.clone(),
            )
            .await
    }
""",
    """    pub fn remove(&mut self, range: Range<usize>) -> bool {
        let Some(range) = Self::checked_range(range) else {
            return false;
        };
        self.ctx.remove(
            self.id,
            range.start,
            range.end,
            self.snapshot.clone(),
        )
    }

    pub async fn remove_async(&mut self, range: Range<usize>) -> bool {
        let Some(range) = Self::checked_range(range) else {
            return false;
        };
        self.ctx
            .remove_async(
                self.id,
                range.start,
                range.end,
                self.snapshot.clone(),
            )
            .await
    }
""",
)

replace_once(
    "src/sequence.rs",
    """        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        self.ctx.copy(
            self.id,
            other.id,
            range.start as i32,
            range.end as i32,
            self.snapshot.clone(),
            other.snapshot.clone(),
        );
""",
    """        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        let range = Self::checked_range(range).expect("sequence range exceeds llama_pos");
        self.ctx.copy(
            self.id,
            other.id,
            range.start,
            range.end,
            self.snapshot.clone(),
            other.snapshot.clone(),
        );
""",
)

replace_once(
    "src/sequence.rs",
    """        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        self.ctx
            .copy_async(
                self.id,
                other.id,
                range.start as i32,
                range.end as i32,
                self.snapshot.clone(),
                other.snapshot.clone(),
            )
            .await;
""",
    """        assert_ne!(self.id, other.id, "cannot copy a sequence onto itself");
        let range = Self::checked_range(range).expect("sequence range exceeds llama_pos");
        self.ctx
            .copy_async(
                self.id,
                other.id,
                range.start,
                range.end,
                self.snapshot.clone(),
                other.snapshot.clone(),
            )
            .await;
""",
)

replace_once(
    "src/sequence.rs",
    """    /// Returns the latest token snapshot.
    pub fn tokens(&self) -> Arc<[Token]> {
        self.snapshot.read().unwrap().tokens.clone()
    }
""",
    """    /// Returns the latest token snapshot.
    pub fn tokens(&self) -> Arc<[Token]> {
        let mut state = self.snapshot.write().unwrap();
        if let Some(tokens) = &state.token_snapshot {
            return tokens.clone();
        }

        let tokens: Arc<[Token]> = Arc::from(state.tokens.as_slice());
        state.token_snapshot = Some(tokens.clone());
        tokens
    }
""",
)

replace_once(
    "tests/sequence.rs",
    "use rusty_llama::{Context, Dist, Greedy, Sampler, Temperature};\n",
    "use rusty_llama::{Context, Dist, Greedy, Sampler, Temperature};\nuse std::sync::Arc;\n",
)

replace_once(
    "tests/sequence.rs",
    """#[test]
fn pos_min_max_after_push() {
""",
    """#[test]
fn remove_rejects_unrepresentable_range() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    assert!(!seq.remove(usize::MAX..usize::MAX));
}

#[test]
#[should_panic(expected = "sequence range exceeds llama_pos")]
fn copy_rejects_unrepresentable_range() {
    let (model, _) = setup();
    let mut params = common::test_ctx_params();
    params.kv_unified = true;
    let ctx = Context::new(&model, &params).unwrap();
    let src = ctx.sequence().unwrap();
    let mut dst = ctx.sequence().unwrap();
    src.copy_to(&mut dst, usize::MAX..usize::MAX);
}

#[test]
fn token_snapshot_is_cached_and_invalidated_by_mutation() {
    let (model, params) = setup();
    let ctx = Context::new(&model, &params).unwrap();
    let mut seq = ctx.sequence().unwrap();
    let token = model.bos_token().unwrap_or(1);

    seq.push(token);
    let first = seq.tokens();
    let cached = seq.tokens();
    assert!(Arc::ptr_eq(&first, &cached));

    seq.push(token);
    let updated = seq.tokens();
    assert!(!Arc::ptr_eq(&first, &updated));
    assert_eq!(first.as_ref(), &[token]);
    assert_eq!(updated.as_ref(), &[token, token]);
}

#[test]
fn pos_min_max_after_push() {
""",
)
