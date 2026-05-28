# Port of examples/simple.rs — greedy argmax generation
#
# Usage:
#   cd packages/elixir && mix run ../../examples/simple.exs --model path/to/model.gguf
#
# Defaults: --prompt "Hello my name is" --n_predict 32 --ngl 99

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
    ngl = opts[:ngl] || 99

    if n_predict <= 0, do: raise("n_predict must be positive, got #{n_predict}")

    {:ok, model} = Model.load_from_file(model_path, ngl)

    IO.puts("Model: #{Model.desc(model)}")

    if Model.has_encoder(model) do
      raise "Model has encoder, which is not supported in this example"
    end

    prompt_tokens = Model.tokenize(model, prompt, true, true)

    {:ok, ctx} = Context.new(model)

    seq = Context.sequence(ctx)

    Sequence.extend(seq, prompt_tokens)

    for token <- prompt_tokens do
      case Model.token_to_piece(model, token) do
        {:ok, piece} -> IO.write(piece)
        _ -> nil
      end
    end

    Enum.reduce_while(1..n_predict, nil, fn _i, _acc ->
      logits = Sequence.logits(seq)

      if is_nil(logits) or logits == [] do
        {:halt, nil}
      else
        {token, _score} =
          logits
          |> Enum.with_index()
          |> Enum.max_by(fn {score, _idx} -> score end)
          |> then(fn {score, idx} -> {idx, score} end)

        if Model.is_eog(model, token) do
          {:halt, nil}
        else
          case Model.token_to_piece(model, token) do
            {:ok, piece} -> IO.write(piece)
            _ -> nil
          end

          Sequence.push(seq, token)
          {:cont, nil}
        end
      end
    end)

    IO.puts("")
  end
end

Simple.main(System.argv())
