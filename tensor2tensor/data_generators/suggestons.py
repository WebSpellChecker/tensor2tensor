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
from tensor2tensor.data_generators.text_encoder import TextEncoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import bleu_hook
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import tokenization

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
        vocab_file = os.path.join(data_dir, 'vocab.txt')
        if(tf.io.gfile.exists(vocab_file)):
            encoder = BERTEncoder(os.path.join(data_dir, 'vocab.txt'))
        else:
            encoder = dotdict({"vocab_size": 28996})
        return encoder

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
        te = text_encoder
        te.PAD = "[PAD]"
        te.SEP = "[SEP]"
        te.CLS = "[CLS]"
        te.DEL = "[unused99]"
        te.EOS = "[unused100]"
        te.RESERVED_TOKENS = [te.PAD, te.SEP, te.CLS, te.DEL, te.EOS]
        te.NUM_RESERVED_TOKENS = len(te.RESERVED_TOKENS)
        te.PAD_ID = 0
        te.SEP_ID = 102
        te.CLS_ID = 101
        te.DEL_ID = 99
        te.EOS_ID = 104
        p.modality = {"targets": modalities.ModalityType.SYMBOL}
        p.vocab_size = {"targets": self._encoders["targets"].vocab_size}
        if self.has_inputs:
            p.modality["inputs"] = modalities.ModalityType.SYMBOL
            p.vocab_size["inputs"] = self._encoders["inputs"].vocab_size
        p.max_length = unused_model_hparams.max_length
        p.max_input_seq_length = unused_model_hparams.max_input_seq_length
        p.max_target_seq_length = unused_model_hparams.max_target_seq_length


class BERTEncoder(TextEncoder):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, vocab_file, num_reserved_ids=0):
        super(BERTEncoder, self).__init__()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):

        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
          s: human-readable string to be converted.

        Returns:
          ids: list of integers
        """
        s_tkn = [text_encoder.CLS] + self.tokenizer.tokenize(s) + [text_encoder.SEP]
        return self.tokenizer.convert_tokens_to_ids(s_tkn)

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
          ids: list of integers to be converted.
          strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).

        Returns:
          s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
          ids: list of integers to be converted.

        Returns:
          strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        return 28996
