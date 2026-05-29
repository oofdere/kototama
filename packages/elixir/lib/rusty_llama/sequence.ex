defmodule RustyLlama.Sequence do
  alias RustyLlama.Native

  def logits(seq), do: Native.sequence_logits(seq)
  def is_empty(seq), do: Native.sequence_is_empty(seq)
  def push(seq, token), do: Native.sequence_push(seq, token)
  def decode(seq), do: Native.sequence_decode(seq)
  def pop(seq), do: Native.sequence_pop(seq)
  def len(seq), do: Native.sequence_len(seq)
  def extend(seq, tokens), do: Native.sequence_extend(seq, tokens)
  def get(seq, index), do: Native.sequence_get(seq, index)
  def pos_min(seq), do: Native.sequence_pos_min(seq)
  def pos_max(seq), do: Native.sequence_pos_max(seq)
  def tokens(seq), do: Native.sequence_tokens(seq)

  # Range args are the half-open token interval [start, stop).
  def remove(seq, start, stop), do: Native.sequence_remove(seq, start, stop)
  def kv_remove(seq, start, stop), do: Native.sequence_kv_remove(seq, start, stop)
  def kv_shift(seq, start, stop, delta), do: Native.sequence_kv_shift(seq, start, stop, delta)
  def copy_to(seq, other, start, stop), do: Native.sequence_copy_to(seq, other, start, stop)
  def copy_from(seq, other, start, stop), do: Native.sequence_copy_from(seq, other, start, stop)
  def kv_copy(seq, other, start, stop), do: Native.sequence_kv_copy(seq, other, start, stop)
end
