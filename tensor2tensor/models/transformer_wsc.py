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
      pad_length = tf.maximum(0, length - max_length)
      sliced = tf.cond(
          tf.less(length, max_length),
          lambda: tf.slice(var, [0, 0], [length, depth]),
          lambda: tf.pad(var, [[0, pad_length], [0, 0]]))
      return x + tf.expand_dims(sliced, 0)
    else:
      return x + tf.gather(var, tf.to_int32(positions))


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
        decoder_input += spans_embd

        if hparams.activation_dtype == "bfloat16":
            decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                                  tf.bfloat16)
        return (decoder_input, decoder_self_attention_bias)

    def get_span_logits(self, attention_list):
        hparams = self.hparams
        attention_list = [
            t for layer_key, t in attention_list.items()
            if "encdec_attention" in layer_key and layer_key.endswith("/logits")
        ]
        attentions = tf.stack(attention_list)
        attentions = tf.reduce_mean(attentions, [0])
        b_size, n_heads, trg_len, inp_len = common_layers.shape_list(attentions)
        pad_length = hparams.max_length - inp_len
        attentions = tf.pad(attentions, [[0, 0], [0, 0], [0, 0], [0, pad_length]])
        span_input = tf.reshape(tf.transpose(attentions, [0, 2, 1, 3]),
                                [b_size, trg_len, hparams.max_length * n_heads])

        span_input = common_layers.layer_preprocess(span_input, hparams)
        span_logits = common_layers.dense(span_input, hparams.max_length, name='span_dense')
        # span_logits = common_layers.layer_postprocess(None, span_logits, hparams)
        return span_logits


    def body(self, features):
        ret = super(TransformerWSC, self).body(features)
        spans = features["spans"]
        span_logits = self.get_span_logits(self.attention_weights)
        """Diagnostic"""
        # print_op = tf.print("span_logits:", span_logits, summarize=10)
        # with tf.control_dependencies([print_op]):
        #     span_logits = span_logits + span_logits - span_logits
        """"""
        span_loss, span_weight = common_layers.padded_cross_entropy(
            span_logits,
            spans,
            self.hparams.label_smoothing)

        span_loss = span_loss / (span_weight + 1e-5)

        return ret, {"span_loss": span_loss}