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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tarfile
import zipfile
from tensor2tensor.data_generators import cleaner_en_xx
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import modalities
from tensor2tensor.utils import bleu_hook
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@registry.register_problem
class Suggestions(text_problems.Text2TextProblem):
    """Base class for translation problems."""

    @property
    def is_generate_per_split(self):
        return True

    @property
    def approx_vocab_size(self):
        return 28996

    @property
    def datatypes_to_clean(self):
        return None

    def source_data_files(self, dataset_split):
        """Files to be passed to compile_data."""
        raise NotImplementedError()

    def vocab_data_files(self):
        """Files to be passed to get_or_generate_vocab."""
        return self.source_data_files(problem.DatasetSplit.TRAIN)

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return dotdict({"vocab_size": 28996})

    def filepattern(self, data_dir, mode, shard=None):
        return "%s/*.tfrecord" % (data_dir)

    def example_reading_spec(self):
        hparams = self._hparams
        data_fields = {
            "src_tokens": tf.FixedLenFeature([hparams.max_input_seq_length], tf.int64),
            "src_tokens_mask": tf.FixedLenFeature([hparams.max_input_seq_length], tf.int64),
            "target_tokens": tf.FixedLenFeature([hparams.max_target_seq_length], tf.int64),
            "target_tokens__mask": tf.FixedLenFeature([hparams.max_target_seq_length], tf.float32),
            "spans": tf.FixedLenFeature([hparams.max_target_seq_length], tf.int64)
        }

        data_items_to_decoders = {
            "inputs": tf.contrib.slim.tfexample_decoder.Tensor("src_tokens"),
            "inputs_mask": tf.contrib.slim.tfexample_decoder.Tensor("src_tokens_mask"),
            "targets": tf.contrib.slim.tfexample_decoder.Tensor('target_tokens'),
            "targets_mask": tf.contrib.slim.tfexample_decoder.Tensor('target_tokens__mask'),
            "spans": tf.contrib.slim.tfexample_decoder.Tensor('spans')
        }
        return (data_fields, data_items_to_decoders)

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        p.modality = {"targets": modalities.ModalityType.SYMBOL}
        p.vocab_size = {"targets": self._encoders["targets"].vocab_size}
        if self.has_inputs:
            p.modality["inputs"] = modalities.ModalityType.SYMBOL
            p.vocab_size["inputs"] = self._encoders["inputs"].vocab_size
        p.max_length = unused_model_hparams.max_length
        p.max_input_seq_length = unused_model_hparams.max_input_seq_length
        p.max_target_seq_length = unused_model_hparams.max_target_seq_length
