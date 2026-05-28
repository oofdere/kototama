# Port of examples/simple.rs — greedy argmax generation
#
# Usage:
#   mix run examples/simple.exs --model path/to/model.gguf --prompt "Hello my name is"
#
# NOTE: ModelParams/ContextParams are not yet exposed through the bindings
# (their C struct fields are invisible to alef's syn parser). This example
# shows the target API shape; params handling will work once rusty_llama
# exports Rust-native param structs.

defmodule Simple do
  alias RustyLlama.{Model, Context, Sequence}

  def main(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [model: :string, prompt: :string, n_predict: :integer, ngl: :integer],
        aliases: [m: :model, n: :n_predict]
      )

    model_path = opts[:model] || raise "missing --model"
    prompt = opts[:prompt] || "Hello my name is"
    n_predict = opts[:n_predict] || 32
    _ngl = opts[:ngl] || 99

    if n_predict <= 0, do: raise("n_predict must be positive, got #{n_predict}")

    # Initialize the model
    # TODO: pass ModelParams once params types are exposed
    {:ok, model} = Model.load_from_file(model_path, %{})

    IO.puts("Model: #{Model.desc(model)}")

    if Model.has_encoder(model) do
      raise "Model has encoder, which is not supported in this example"
    end

    # Tokenize the prompt
    prompt_tokens = Model.tokenize(model, prompt, true, true)

    # Initialize the context
    # TODO: pass ContextParams once params types are exposed
    {:ok, ctx} = Context.new(model, %{})

    seq = Context.sequence(ctx)

    Sequence.extend(seq, prompt_tokens)

    # Print the prompt token-by-token
    for token <- prompt_tokens do
      IO.write(Model.token_to_piece(model, token))
    end

    # Main loop — greedy argmax sampling
    Enum.reduce_while(1..n_predict, nil, fn _i, _acc ->
      logits = Sequence.logits(seq)

      {token, _score} =
        logits
        |> Enum.with_index()
        |> Enum.max_by(fn {score, _idx} -> score end)
        |> then(fn {score, idx} -> {idx, score} end)

      if Model.is_eog(model, token) do
        {:halt, nil}
      else
        IO.write(Model.token_to_piece(model, token))
        Sequence.push(seq, token)
        {:cont, nil}
      end
    end)

    IO.puts("")
  end
end

Simple.main(System.argv())
