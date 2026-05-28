# Port of examples/simple_chat.rs — interactive chat loop
#
# Usage:
#   mix run examples/simple_chat.exs --model path/to/model.gguf
#
# NOTE: ModelParams/ContextParams/SamplerChain are not yet exposed through
# the bindings. This example shows the target API shape using greedy
# sampling as a stand-in for the sampler chain.

defmodule SimpleChat do
  alias RustyLlama.{Model, Context, Sequence}

  def main(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [model: :string, context: :integer, ngl: :integer],
        aliases: [m: :model, c: :context]
      )

    model_path = opts[:model] || raise "missing --model"
    _context_size = opts[:context] || 2048
    _ngl = opts[:ngl] || 99

    # Initialize the model
    # TODO: pass ModelParams with n_gpu_layers once params are exposed
    {:ok, model} = Model.load_from_file(model_path, %{})

    # Initialize the context
    # TODO: pass ContextParams with n_ctx/n_batch once params are exposed
    {:ok, ctx} = Context.new(model, %{})

    seq = Context.sequence(ctx)

    chat_loop(model, seq, [])
  end

  defp chat_loop(model, seq, messages) do
    input = IO.gets("") |> String.trim()

    if input == "" do
      :ok
    else
      messages = messages ++ [{:user, input}]

      prompt = format_messages(messages)
      is_first = Sequence.is_empty(seq)

      tokens = Model.tokenize(model, prompt, is_first, true)
      Sequence.extend(seq, tokens)

      IO.puts("")

      {response, _} = generate_response(model, seq)

      IO.puts("")

      messages = messages ++ [{:assistant, response}]
      chat_loop(model, seq, messages)
    end
  end

  defp generate_response(model, seq, response \\ "") do
    logits = Sequence.logits(seq)

    # Greedy argmax as stand-in for SamplerChain (min_p + temp + dist)
    {token, _score} =
      logits
      |> Enum.with_index()
      |> Enum.max_by(fn {score, _idx} -> score end)
      |> then(fn {score, idx} -> {idx, score} end)

    if Model.is_eog(model, token) do
      {response, seq}
    else
      piece = Model.token_to_piece(model, token)
      IO.write(piece)
      Sequence.push(seq, token)

      response = response <> piece

      if String.contains?(piece, "\n") do
        {response, seq}
      else
        generate_response(model, seq, response)
      end
    end
  end

  defp format_messages(messages) do
    formatted =
      Enum.map_join(messages, "\n", fn
        {:user, text} -> "user: #{text}"
        {:assistant, text} -> "assistant: #{text}"
      end)

    result = formatted <> "\nassistant:"
    IO.puts(result)
    result
  end
end

SimpleChat.main(System.argv())
