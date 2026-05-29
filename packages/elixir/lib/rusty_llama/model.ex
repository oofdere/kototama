defmodule RustyLlama.Model do
  alias RustyLlama.Native

  def load_from_file(path, n_gpu_layers \\ 99) do
    Native.model_load_from_file(path, n_gpu_layers)
  end

  def chat_template(model, name \\ nil), do: Native.model_chat_template(model, name)
  def desc(model), do: Native.model_desc(model)
  def has_decoder(model), do: Native.model_has_decoder(model)
  def decoder_start_token(model), do: Native.model_decoder_start_token(model)
  def has_encoder(model), do: Native.model_has_encoder(model)
  def is_diffusion(model), do: Native.model_is_diffusion(model)
  def is_hybrid(model), do: Native.model_is_hybrid(model)
  def is_recurrent(model), do: Native.model_is_recurrent(model)
  def token_to_piece(model, token), do: Native.model_token_to_piece(model, token)
  def tokenize(model, text, add_special \\ true, parse_special \\ true), do: Native.model_tokenize(model, text, add_special, parse_special)
  def get_add_bos(model), do: Native.model_get_add_bos(model)
  def get_add_eos(model), do: Native.model_get_add_eos(model)
  def get_add_sep(model), do: Native.model_get_add_sep(model)
  def get_score(model, token), do: Native.model_get_score(model, token)
  def get_text(model, token), do: Native.model_get_text(model, token)
  def is_control(model, token), do: Native.model_is_control(model, token)
  def is_eog(model, token), do: Native.model_is_eog(model, token)
  def n_tokens(model), do: Native.model_n_tokens(model)

  # Special-token getters. Each returns the token id, or nil if the vocab
  # does not define it.
  def bos_token(model), do: Native.model_bos_token(model)
  def cls_token(model), do: Native.model_cls_token(model)
  def eos_token(model), do: Native.model_eos_token(model)
  def eot_token(model), do: Native.model_eot_token(model)
  def fim_mid_token(model), do: Native.model_fim_mid_token(model)
  def fim_pad_token(model), do: Native.model_fim_pad_token(model)
  def fim_pre_token(model), do: Native.model_fim_pre_token(model)
  def fim_rep_token(model), do: Native.model_fim_rep_token(model)
  def fim_sep_token(model), do: Native.model_fim_sep_token(model)
  def fim_suf_token(model), do: Native.model_fim_suf_token(model)
  def mask_token(model), do: Native.model_mask_token(model)
  def nl_token(model), do: Native.model_nl_token(model)
  def pad_token(model), do: Native.model_pad_token(model)
  def sep_token(model), do: Native.model_sep_token(model)
end
