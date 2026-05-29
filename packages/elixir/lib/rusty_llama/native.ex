defmodule RustyLlama.Native do
  use Rustler,
    otp_app: :rusty_llama,
    crate: "rusty_llama_nif",
    path: "native/rusty_llama_nif"

  # Model
  def model_load_from_file(_path, _n_gpu_layers), do: :erlang.nif_error(:nif_not_loaded)
  def model_chat_template(_model, _name), do: :erlang.nif_error(:nif_not_loaded)
  def model_desc(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_has_decoder(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_decoder_start_token(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_has_encoder(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_is_diffusion(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_is_hybrid(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_is_recurrent(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_token_to_piece(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def model_tokenize(_model, _text, _add_special, _parse_special), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_add_bos(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_add_eos(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_add_sep(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_score(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_text(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def model_is_control(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def model_is_eog(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def model_n_tokens(_model), do: :erlang.nif_error(:nif_not_loaded)

  # Context
  def context_new(_model, _n_ctx, _n_batch), do: :erlang.nif_error(:nif_not_loaded)
  def context_sequence(_context), do: :erlang.nif_error(:nif_not_loaded)
  def context_free_slots(_context), do: :erlang.nif_error(:nif_not_loaded)
  def context_n_ctx(_context), do: :erlang.nif_error(:nif_not_loaded)
  def context_can_shift(_context), do: :erlang.nif_error(:nif_not_loaded)

  # Sequence
  def sequence_logits(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_is_empty(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_push(_sequence, _token), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_decode(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_pop(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_len(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_extend(_sequence, _tokens), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_get(_sequence, _index), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_pos_min(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_pos_max(_sequence), do: :erlang.nif_error(:nif_not_loaded)
  def sequence_tokens(_sequence), do: :erlang.nif_error(:nif_not_loaded)
end
