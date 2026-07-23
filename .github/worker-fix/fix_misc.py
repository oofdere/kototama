from pathlib import Path

Path("src/sequence.rs").write_text(Path(".github/worker-fix/sequence.rs").read_text())

sequence_tests = Path("tests/sequence.rs")
s = sequence_tests.read_text()
s = s.replace("assert_eq!(seq.tokens(), tokens.as_slice());", "assert_eq!(seq.tokens().as_ref(), tokens.as_slice());")
s = s.replace("fn index_operator() {", "fn token_snapshot_supports_indexing() {")
s = s.replace("assert_eq!(seq[0], tokens[0]);", "assert_eq!(seq.tokens()[0], tokens[0]);")
s = s.replace("assert_eq!(dst.tokens(), src.tokens());", "assert_eq!(dst.tokens().as_ref(), src.tokens().as_ref());")
s = s.replace("assert_eq!(seq1.tokens(), tokens1.as_slice());", "assert_eq!(seq1.tokens().as_ref(), tokens1.as_slice());")
s = s.replace("assert_eq!(seq2.tokens(), tokens2.as_slice());", "assert_eq!(seq2.tokens().as_ref(), tokens2.as_slice());")
s = s.replace("assert_ne!(seq1.tokens(), seq2.tokens());", "assert_ne!(seq1.tokens().as_ref(), seq2.tokens().as_ref());")
s = s.replace("assert_ne!(seq1.logits().unwrap(), seq2.logits().unwrap());", "assert_ne!(\n        seq1.logits().unwrap().as_ref(),\n        seq2.logits().unwrap().as_ref()\n    );")
s = s.replace("let l = temp.apply(logits);", "let l = temp.apply(&logits);")
sequence_tests.write_text(s)

integration = Path("tests/integration.rs")
s = integration.read_text()
s = s.replace("let l = temp.apply(logits);", "let l = temp.apply(&logits);")
s = s.replace("assert_eq!(seq_a.tokens(), tokens_a.as_slice());", "assert_eq!(seq_a.tokens().as_ref(), tokens_a.as_slice());")
s = s.replace("assert_eq!(seq_b.tokens(), tokens_b.as_slice());", "assert_eq!(seq_b.tokens().as_ref(), tokens_b.as_slice());")
s = s.replace("        seq_a.logits().unwrap(),\n        seq_b.logits().unwrap(),", "        seq_a.logits().unwrap().as_ref(),\n        seq_b.logits().unwrap().as_ref(),")
integration.write_text(s)

worker = Path("tests/worker.rs")
s = worker.read_text()
s = s.replace("assert_eq!(seq.tokens(), tokens.as_slice());", "assert_eq!(seq.tokens().as_ref(), tokens.as_slice());")
if "dropping_sequence_checkout_future_releases_the_slot" not in s:
    s += """

#[test]
fn dropping_sequence_checkout_future_releases_the_slot() {
    let (model, params) = common::load_model_and_context();
    let ctx = Context::new(&model, &params).unwrap();
    let initial_slots = ctx.free_slots();

    let checkout = ctx.sequence_async();
    drop(checkout);

    // free_slots is submitted after the cancellation command, so its reply is
    // also a worker barrier for the abandoned checkout.
    assert_eq!(ctx.free_slots(), initial_slots);
}
"""
worker.write_text(s)

chat = Path("examples/simple_chat.rs")
s = chat.read_text().replace("let l = minp.apply(logits);", "let l = minp.apply(&logits);")
chat.write_text(s)
