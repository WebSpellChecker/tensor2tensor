# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow_probability as tfp

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers import transformer_layers
from tensor2tensor.layers import transformer_memory
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.utils import t2t_model
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest

# pylint: enable=g-direct-tensorflow-import

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def add_positional_embedding(x, max_length, pos_embd, positions=None):
    _, length, depth = common_layers.shape_list(x)
    tf.cast(pos_embd, x.dtype)
    var = pos_embd
    if positions is None:
      # pad_length = tf.maximum(0, length - max_length)
      sliced = tf.slice(var, [0, 0], [length, depth])
      return x + tf.expand_dims(sliced, 0)
    else:
      return x + tf.gather(var, tf.to_int32(positions))


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def _init_transformer_cache(cache, hparams, batch_size, attention_init_length,
                            encoder_output, encoder_decoder_attention_bias,
                            scope_prefix):
  """Create the initial cache for Transformer fast decoding."""
  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if cache is None:
    cache = {}
  cache.update({
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            key_channels]), hparams.num_heads),
          "v":
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            value_channels]), hparams.num_heads),
      } for layer in range(num_layers)
  })

  # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
  # cache key "f" won't be used, which means that the` shape of cache["f"]`
  # won't be changed to
  # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
  # error when applying `nest.map reshape function` on it.
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" %
          (scope_prefix, layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
  return cache

@registry.register_model
class TransformerWSC(transformer.Transformer):
    """Attention net.  See file docstring."""

    def __init__(self, *args, **kwargs):
        super(TransformerWSC, self).__init__(*args, **kwargs)
        self._prepare_encoder_fn = self.prepare_encoder_fn
        self._prepare_decoder_fn = self.prepare_decoder_fn
        with tf.name_scope("positional_embedding"):
            self.pos_embd = tf.get_variable("inputs_positional_embedding",
                                            [self.hparams.max_length, self.hparams.hidden_size])
        #if


    def prepare_encoder_fn(self, inputs, target_space, hparams, features=None,
                           type_ids=None, num_types=None):
        """Prepare one shard of the model for the encoder.

        Args:
          inputs: a Tensor.
          target_space: a Tensor.
          hparams: run hyperparameters
          features: optionally pass the entire features dictionary as well.
            This is needed now for "packed" datasets.
          type_ids: optional, an int64 Tensor of shape [batch, length] that allows
            for adding type embeddings, similar to positional embeddings.
          num_types: optional, an int that decides the number of types in type_ids.

        Returns:
          encoder_input: a Tensor, bottom of encoder stack
          encoder_self_attention_bias: a bias tensor for use in encoder self-attention
          encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
            attention
        """
        ishape_static = inputs.shape.as_list()
        encoder_input = inputs
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        if (hasattr(hparams, "unidirectional_encoder") and
                hparams.unidirectional_encoder):
            tf.logging.info("Using unidirectional encoder")
            encoder_self_attention_bias = (
                common_attention.attention_bias_lower_triangle(
                    common_layers.shape_list(inputs)[1]))
        else:
            # Usual case - not a packed dataset.
            encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        inputs_position = None
        if hparams.proximity_bias:
            encoder_self_attention_bias += common_attention.attention_bias_proximal(
                common_layers.shape_list(inputs)[1])

        encoder_input = add_positional_embedding(encoder_input, self.hparams.max_length, self.pos_embd, inputs_position)

        encoder_self_attention_bias = common_layers.cast_like(
            encoder_self_attention_bias, encoder_input)
        encoder_decoder_attention_bias = common_layers.cast_like(
            encoder_decoder_attention_bias, encoder_input)
        return (encoder_input, encoder_self_attention_bias,
                encoder_decoder_attention_bias)


    def prepare_decoder_fn(self, targets, hparams, features=None, pad=None):
        """Prepare one shard of the model for the decoder.

        Args:
          targets: a Tensor.
          hparams: run hyperparameters
          features: optionally pass the entire features dictionary as well. This is
            needed now for "packed" datasets.
          pad: vector to use for padding when shifting targets right

        Returns:
          decoder_input: a Tensor, bottom of decoder stack
          decoder_self_attention_bias: a bias tensor for use in decoder self-attention
        """
        if hparams.causal_decoder_self_attention:
            # Causal attention.
            if hparams.prepend_mode == "prepend_inputs_full_attention":
                decoder_self_attention_bias = (
                    common_attention.attention_bias_prepend_inputs_full_attention(
                        common_attention.embedding_to_padding(targets)))
            else:
                decoder_self_attention_bias = (
                    common_attention.attention_bias_lower_triangle(
                        common_layers.shape_list(targets)[1]))
        else:
            # Full attention.
            decoder_padding = common_attention.embedding_to_padding(targets)
            decoder_self_attention_bias = (
                common_attention.attention_bias_ignore_padding(decoder_padding))

        if features and "targets_segmentation" in features:
            # "Packed" dataset - keep the examples from seeing each other.
            targets_segmentation = features["targets_segmentation"]
            if 'targets_position' in features:
                targets_position = features["targets_position"]
            else:
                targets_position = None
            decoder_self_attention_bias += common_attention.attention_bias_same_segment(
                targets_segmentation, targets_segmentation)
        else:
            targets_position = None
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                common_layers.shape_list(targets)[1])
        decoder_input = common_layers.shift_right_3d(targets, pad)

        if targets_position is not None:
            decoder_input = common_attention.add_timing_signal_1d_given_position(
                decoder_input, targets_position)
        else:
            decoder_input = common_attention.add_timing_signal_1d(decoder_input)
        spans = features["spans"]
        pos_embd = tf.cast(self.pos_embd, decoder_input.dtype)
        spans_sign = tf.cast(tf.expand_dims(tf.sign(spans), [-1]),  decoder_input.dtype)
        spans_embd = tf.gather(pos_embd, tf.to_int32(spans))
        spans_embd = spans_embd * spans_sign
        #Pad like decoder_input
        spans_embd = common_layers.shift_right_3d(spans_embd, pad)
        decoder_input += spans_embd

        if hparams.activation_dtype == "bfloat16":
            decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                                  tf.bfloat16)
        return (decoder_input, decoder_self_attention_bias)

    def get_span_logits(self, attention_list):
        hparams = self.hparams
        attention_list = [
            t for layer_key, t in attention_list.items()
            if "encdec_attention" in layer_key and (not layer_key.endswith("/logits"))
        ]
        attentions = tf.stack(attention_list)
        attentions = attentions[-1]

        b_size, n_heads, trg_len, inp_len = common_layers.shape_list(attentions)
        pad_length = hparams.max_length - inp_len
        attentions = tf.pad(attentions, [[0, 0], [0, 0], [0, 0], [0, pad_length]])
        span_input = tf.reshape(tf.transpose(attentions, [0, 2, 1, 3]),
                                [b_size, trg_len, hparams.max_length * self.hparams.num_heads])
        span_input = common_layers.layer_preprocess(span_input, hparams)
        span_logits = common_layers.dense(span_input, hparams.max_length, name='span_dense')
        # span_logits = common_layers.layer_postprocess(None, span_logits, hparams)
        return span_logits

    def get_decode_end_id(self):
        return text_encoder.EOS_ID

    def body(self, features):
        ret = super(TransformerWSC, self).body(features)
        spans = features["spans"]
        span_logits = self.get_span_logits(self.attention_weights)
        print('BATCH SIZE', span_logits.shape[0])
        span_loss, span_weight = common_layers.padded_cross_entropy(
            span_logits,
            spans,
            self.hparams.label_smoothing)

        span_loss = span_loss / (span_weight + 1e-5)

        return ret, {"span_loss": span_loss}


    def _beam_decode(self,
                     features,
                     decode_length,
                     beam_size,
                     top_beams,
                     alpha,
                     use_tpu=False):
        """Beam search decoding.
        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
        """
        beam_size = 1
        with tf.variable_scope(self.name, default_name="embedding"):
            return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha, use_tpu)

    def _fast_decode_tpu(self,
                         features,
                         decode_length,
                         beam_size=1,
                         top_beams=1,
                         alpha=1.0,
                         use_tpu=False):
        """Fast decoding.

        Implements both greedy and beam search decoding on TPU, uses beam search
        iff beam_size > 1, otherwise beam search related arguments are ignored.

        Args:
          features: A map of string to model features.
          decode_length: An integer, how many additional timesteps to decode.
          beam_size: An integer, number of beams.
          top_beams: An integer, how many of the beams to return.
          alpha: A float that controls the length penalty. Larger the alpha,
            stronger the preference for longer translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }.

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]
        target_vocab_size = self._problem_hparams.vocab_size["targets"]
        if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
            target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

        inputs = features["inputs"]
        features["inputs"] = inputs
        inputs_shape = common_layers.shape_list(inputs)
        batch_size = inputs_shape[0]
        inputs = self._prepare_inputs_for_decode(features)

        with tf.variable_scope("body"):
            encoder_output, encoder_decoder_attention_bias = dp(
                self.encode,
                inputs,
                features["target_space_id"],
                hparams,
                features=features)
        encoder_output = encoder_output[0]

        encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
        partial_targets = None
        sep_id = text_encoder.SEP_ID

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

        def preprocess_targets(targets, span, is_span, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: A tensor, inputs ids to the decoder. [batch_size, 1].
              i: An integer, Step number of the decoding loop.

            Returns:
              A tensor, processed targets [batch_size, 1, hidden_dim].
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            modality_name = hparams.name.get(
                "targets",
                modalities.get_name(target_modality))(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                bottom = hparams.bottom.get(
                    "targets", modalities.get_targets_bottom(target_modality))
                targets = dp(bottom, targets, hparams, target_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)

            # GO embeddings are all zero, this is because transformer_prepare_decoder
            # Shifts the targets along by one for the input which pads with zeros.
            # If the modality already maps GO to the zero embeddings this is not
            # needed.
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                positional_encoding_shape = positional_encoding.shape.as_list()
                targets += tf.slice(
                    positional_encoding, [0, i, 0],
                    [positional_encoding_shape[0], 1, positional_encoding_shape[2]])

            pos_embd = tf.cast(self.pos_embd, targets.dtype)
            span_embd = tf.gather(pos_embd, tf.to_int32(span))
            span_embd *= tf.expand_dims(tf.cast(is_span, dtype=span_embd.dtype), axis=[-1])
            targets += span_embd

            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)
        beam_batch_size = beam_size * batch_size

        def symbols_to_logits_tpu_fn(ids, i, cache):
            """Go from ids to logits for next symbol on TPU.

            Args:
              ids: A tensor, symbol IDs.
              i: An integer, step number of the decoding loop. Only used for inference
                on TPU.
              cache: A dict, containing tensors which are the results of previous
                attentions, used for fast decoding.

            Returns:
              ret: A tensor, computed logits.
              cache: A dict, containing tensors which are the results of previous
                  attentions, used for fast decoding.
            """
            seq, spans = cache['seq'], cache['spans']
            ids = tf.reshape(ids[:, -1:], [-1, 1])
            before_ids = tf.slice(seq, [0, i], [beam_batch_size, 1])
            is_span = tf.logical_or(tf.equal(ids, sep_id), tf.equal(before_ids, sep_id))
            is_span = tf.logical_or(is_span, tf.equal(i * tf.ones_like(ids), 1))
            is_span = tf.cast(is_span, dtype=tf.int32)

            span = tf.slice(spans, [0, i], [beam_batch_size, 1])
            spans = tf.transpose(spans)
            spans = inplace_ops.alias_inplace_update(
                spans, i, tf.squeeze(span * is_span, axis=1))
            spans = tf.transpose(spans)
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)

            targets = preprocess_targets(targets, span, is_span, i)

            bias_shape = decoder_self_attention_bias.shape.as_list()
            bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    i,
                    nonpadding=features_to_nonpadding(features, "targets"))

            modality_name = hparams.name.get(
                "targets",
                modalities.get_name(target_modality))(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                top = hparams.top.get("targets",
                                      modalities.get_top(target_modality))
                logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]
            ret = tf.squeeze(logits, axis=[1, 2, 3])
            """Spans proccesing."""
            with tf.variable_scope("body"):
                span_logits = self.get_span_logits(self.attention_weights)
                next_span = tf.argmax(tf.nn.softmax(span_logits, axis=-1), axis=-1, output_type=tf.int32)
                next_span = tf.reshape(next_span, [beam_batch_size, 1])
                spans = tf.transpose(spans)
                spans = inplace_ops.alias_inplace_update(
                    spans, i + 1, tf.squeeze(next_span, axis=1))
                spans = tf.transpose(spans)

            """"""
            seq = tf.transpose(seq)
            seq = inplace_ops.alias_inplace_update(
                seq, i + 1, tf.squeeze(ids, axis=1))
            seq = tf.transpose(seq)
            cache['seq'], cache['spans'] = seq, spans
            return ret, cache

        eos_id = self.get_decode_end_id() or beam_search.EOS_ID
        ret = fast_decode_tpu(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_tpu_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_vocab_size,
            init_cache_fn=self._init_cache_fn,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length,
            eos_id=eos_id,
            use_tpu=use_tpu)

        return ret

    def estimator_spec_predict(self, features, use_tpu=False):
        """Constructs `tf.estimator.EstimatorSpec` for PREDICT (inference) mode."""
        decode_hparams = self._decode_hparams
        top_beams = decode_hparams.beam_size if decode_hparams.return_beams else 1
        infer_out = self.infer(
            features,
            beam_size=decode_hparams.beam_size,
            top_beams=top_beams,
            alpha=decode_hparams.alpha,
            decode_length=decode_hparams.extra_length,
            use_tpu=use_tpu)
        if isinstance(infer_out, dict):
            outputs = infer_out["outputs"]
            scores = infer_out["scores"]
            spans = infer_out["spans"]
        else:
            outputs = infer_out
            scores = None

        # Workaround for "ValueError: prediction values must be from the default
        # graph" during TPU model exporting.
        # TODO(b/130501786): remove tf.identity once default graph mismatch is fixed
        for name, feature in features.items():
            features[name] = tf.identity(feature)

        inputs = features.get("inputs")
        if inputs is None:
            inputs = features["targets"]

        predictions = {
            "outputs": outputs,
            "scores": scores,
            "inputs": inputs,
            "targets": features.get("infer_targets"),
            "spans": spans
        }

        # Pass through remaining features
        for name, feature in features.items():
            if name not in list(predictions.keys()) + ["infer_targets"]:
                if name == "decode_loop_step":
                    continue
                if not feature.shape.as_list():
                    # All features must have a batch dimension
                    batch_size = common_layers.shape_list(outputs)[0]
                    feature = tf.tile(tf.expand_dims(feature, 0), [batch_size])
                predictions[name] = feature

            t2t_model._del_dict_non_tensors(predictions)

        export_out = {"outputs": predictions["outputs"]}
        if "scores" in predictions:
            export_out["scores"] = predictions["scores"]

        # Necessary to rejoin examples in the correct order with the Cloud ML Engine
        # batch prediction API.
        if "batch_prediction_key" in predictions:
            export_out["batch_prediction_key"] = predictions["batch_prediction_key"]

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(export_out)
        }
        if use_tpu:
            # Note: important to call this before remove_summaries()
            if self.hparams.tpu_enable_host_call:
                host_call = self.create_eval_host_call()
            else:
                host_call = None

            t2t_model.remove_summaries()

            return tf.contrib.tpu.TPUEstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                host_call=host_call,
                export_outputs=export_outputs)
        else:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs=export_outputs)

def fast_decode_tpu(encoder_output,
                    encoder_decoder_attention_bias,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    init_cache_fn=_init_transformer_cache,
                    beam_size=1,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    force_decode_length=False,
                    scope_prefix="body/",
                    use_top_k_with_unique=True,
                    use_tpu=False):
    """Given encoder output and a symbols to logits function, does fast decoding.

    Implements both greedy and beam search decoding for TPU, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      encoder_output: A tensor, output from encoder.
      encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
        attention.
      symbols_to_logits_fn: Incremental decoding, function mapping triple `(ids,
        step, cache)` to symbol logits.
      hparams: Run hyperparameters.
      decode_length: An integer, how many additional timesteps to decode.
      vocab_size: Output vocabulary size.
      init_cache_fn: Function that returns the initial cache dict.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha, stronger
        the preference for longer translations.
      sos_id: Start-of-sequence symbol.
      eos_id: End-of-sequence symbol.
      batch_size: An integer, must be passed if there is no input.
      force_decode_length: A bool, whether to force the full decode length, or if
        False, stop when all beams hit eos_id.
      scope_prefix: str, prefix for decoder layer variable scopes.
      use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
        top_k during beam search.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
    """


    cache = init_cache_fn(None, hparams, batch_size, decode_length,
                          encoder_output, encoder_decoder_attention_bias,
                          scope_prefix)
    cache["encoder_output"] = encoder_output
    cache["spans"] = tf.zeros([batch_size, decode_length + 1], dtype=tf.int32)
    cache["seq"] = tf.zeros([batch_size, decode_length + 1], dtype=tf.int32)

    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
        value={
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "beam_size": beam_size,
            "alpha": alpha,
            "max_decode_length": decode_length
        },
        hparams=hparams)
    if beam_size > 1:  # Beam Search
        initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
        decoded_ids, scores, res_cache = beam_search.beam_search(
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states=cache,
            eos_id=eos_id,
            stop_early=(top_beams == 1),
            use_tpu=use_tpu,
            use_top_k_with_unique=use_top_k_with_unique)

        spans = tf.reshape(res_cache["spans"], [batch_size, beam_size, decode_length + 1])
        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
            spans = spans[:, 0, :]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
            scores = scores[:, :top_beams]
            spans = spans[:, :top_beams, 1:]
    else:
        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = getattr(hparams, "sampling_temp", 0.0)
            keep_top = getattr(hparams, "sampling_keep_top_k", -1)
            if hparams.sampling_method == "argmax":
                temperature = 0.0
            next_id = common_layers.sample_with_temperature(
                logits, temperature, keep_top)
            next_id = tf.cast(next_id, dtype=tf.int32)

            log_prob_indices = tf.stack([tf.range(tf.to_int32(batch_size)), next_id],
                                        axis=1)
            log_prob += tf.gather_nd(
                log_probs, log_prob_indices) * (1 - tf.to_float(hit_eos))
            # Note(thangluong): we purposely update hit_eos after aggregating log_prob
            # There is a subtle detail here that we want to include log_probs up to
            # (and inclusive of) the first eos generated, but not subsequent tokens.
            hit_eos |= tf.equal(next_id, eos_id)

            next_id = tf.expand_dims(next_id, axis=1)
            decoded_ids = tf.transpose(decoded_ids)
            decoded_ids = inplace_ops.alias_inplace_update(
                decoded_ids, i, tf.squeeze(next_id, axis=1))
            decoded_ids = tf.transpose(decoded_ids)
            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int32)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int32)
        initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)

        def compute_cache_shape_invariants(tensor):
            return tf.TensorShape(tensor.shape.as_list())

        bs_shape = batch_size if isinstance(batch_size, int) else None
        _, _, _, decoded_ids, res_cache, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop, [
                tf.constant(0), hit_eos, next_id, decoded_ids, cache,
                initial_log_prob
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([bs_shape]),
                tf.TensorShape([bs_shape, 1]),
                tf.TensorShape([bs_shape, decode_length]),
                nest.map_structure(compute_cache_shape_invariants, cache),
                tf.TensorShape([bs_shape]),
            ])
        spans = tf.reshape(res_cache["spans"], [batch_size, decode_length + 1])
        spans = spans[:, 1:]
        scores = log_prob
    return {"outputs": decoded_ids, "scores": scores, "spans": spans}
