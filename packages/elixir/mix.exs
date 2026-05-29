defmodule RustyLlama.MixProject do
  use Mix.Project

  def project do
    [
      app: :rusty_llama,
      version: "0.1.0",
      elixir: "~> 1.14",
      description: "Elixir bindings for rusty-llama (llama.cpp)",
      package: package(),
      deps: deps()
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/oofdere/rusty_llama"},
      files:
        ~w(.formatter.exs mix.exs README* native/rusty_llama_nif/Cargo.toml native/rusty_llama_nif/Cargo.lock native/rusty_llama_nif/src)
    ]
  end

  defp deps do
    [
      {:rustler, "~> 0.37.0"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end
end
