defmodule RustyLlama.Context do
  alias RustyLlama.Native

  def new(model, n_ctx \\ 2048, n_batch \\ 512) do
    Native.context_new(model, n_ctx, n_batch)
  end

  def sequence(context), do: Native.context_sequence(context)
  def free_slots(context), do: Native.context_free_slots(context)
  def n_ctx(context), do: Native.context_n_ctx(context)
  def can_shift(context), do: Native.context_can_shift(context)
end
