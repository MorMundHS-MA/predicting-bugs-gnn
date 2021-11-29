from typing import NamedTuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPool1D

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1
node_label_embedding_size = 32
language = "java"
graph_node_label_max_num_chars = 19


class LabelToNodeRepresentationInput(NamedTuple):
    """A named tuple to hold input for the CharCNN label representation"""
    unique_labels_as_characters: tf.Tensor
    node_labels_to_unique_labels: tf.Tensor


class NodeLabelCharCNNEmbeddingLayer(keras.layers.Layer):
    def __init__(self, graph_node_label_representation_size: int):
        super(NodeLabelCharCNNEmbeddingLayer, self).__init__()
        self.graph_node_label_representation_size = graph_node_label_representation_size

        # Choose kernel sizes such that there is a single value at the end:
        char_conv_l1_kernel_size = 5
        char_conv_l2_kernel_size = \
            graph_node_label_max_num_chars - 2 * (char_conv_l1_kernel_size - 1)

        self._char_conv_l1 = Conv1D(filters=16,
                                    kernel_size=char_conv_l1_kernel_size,
                                    activation=tf.nn.leaky_relu,
                                    )  # Shape: [U, C - (char_conv_l1_kernel_size - 1), 16]
        self._char_pool_l1 = MaxPool1D(pool_size=char_conv_l1_kernel_size,
                                       strides=1,
                                       )  # Shape: [U, C - 2*(char_conv_l1_kernel_size - 1), 16]
        self._char_conv_l2 = Conv1D(filters=self.graph_node_label_representation_size,
                                    kernel_size=char_conv_l2_kernel_size,
                                    activation=tf.nn.leaky_relu,
                                    )  # Shape: [U, 1, D]

    def call(self, inputs: LabelToNodeRepresentationInput, **kwargs):
        unique_labels_as_characters = inputs.unique_labels_as_characters
        node_labels_to_unique_labels = inputs.node_labels_to_unique_labels

        # U ~ num unique labels
        # C ~ num characters (self.params['graph_node_label_max_num_chars'])
        # A ~ num characters in alphabet
        unique_label_chars_one_hot = tf.one_hot(indices=unique_labels_as_characters,
                                                depth=len(ALPHABET),
                                                axis=-1)  # Shape: [U, C, A]

        char_conv_l1 = self._char_conv_l1(unique_label_chars_one_hot)
        char_pool_l1 = self._char_pool_l1(inputs=char_conv_l1)
        char_conv_l2 = self._char_conv_l2(char_pool_l1)

        unique_label_representations = tf.squeeze(char_conv_l2, axis=1)  # Shape: [U, D]
        node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=node_labels_to_unique_labels)
        return node_label_representations
