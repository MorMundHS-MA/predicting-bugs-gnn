import typing
from typing import Tuple, List, Dict

import tensorflow as tf

from models.source_graph.source_graph_input import SourceGraphInput
from models.source_graph.string2tensor import String2Tensor
from models.source_graph.source_graph_tensors import merge_samples_with_graph_node_offset


class SourceGraphDiffInput(SourceGraphInput):
    _program_graph_edge_types = ["Child", "NextToken", "LastUsedVariable", "LastUse", "LastWrite", "LastLexicalUse",
                                 "ComputedFrom", "ReturnsTo", "FormalArgName", "GuardedBy",
                                 "GuardedByNegation", "BindsToSymbol"]

    def __init__(self, graphs_in_batch: tf.Tensor, node_labels_unique_a: tf.Tensor,
                 node_labels_index_a: tf.Tensor, node_to_graph_a: tf.Tensor,
                 adjacency_lists_a: Tuple[tf.Tensor],
                 node_labels_unique_b: tf.Tensor,
                 node_labels_index_b: tf.Tensor, node_to_graph_b: tf.Tensor,
                 adjacency_lists_b: Tuple[tf.Tensor]):
        self.graphs_in_batch = graphs_in_batch
        self.node_labels_unique_a = node_labels_unique_a
        self.node_labels_index_a = node_labels_index_a
        self.node_to_graph_a = node_to_graph_a
        self.adjacency_lists_a = adjacency_lists_a
        self.node_labels_unique_b = node_labels_unique_b
        self.node_labels_index_b = node_labels_index_b
        self.node_to_graph_b = node_to_graph_b
        self.adjacency_lists_b = adjacency_lists_b

    _string2tensor = String2Tensor.get_default()
    _adjacency_type = tf.int32
    _spec_graphs_in_batch = tf.TensorSpec(shape=(), dtype=tf.int32)
    _spec_node_labels_unique_a = tf.TensorSpec(shape=(None, _string2tensor.get_node_label_max_chars()), dtype=tf.uint8)
    _spec_node_labels_index_a = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    _spec_node_to_graph_a = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    _spec_adjacency_lists_a = [tf.TensorSpec((None, 2), dtype=_adjacency_type)] * len(_program_graph_edge_types)
    _spec_node_labels_unique_b = _spec_node_labels_unique_a
    _spec_node_labels_index_b = _spec_node_labels_index_a
    _spec_node_to_graph_b = _spec_node_to_graph_a
    _spec_adjacency_lists_b = _spec_adjacency_lists_a

    _spec = [
        _spec_graphs_in_batch,
        _spec_node_labels_unique_a,
        _spec_node_labels_index_a,
        _spec_node_to_graph_a,
        _spec_node_labels_unique_b,
        _spec_node_labels_index_b,
        _spec_node_to_graph_b,
        *_spec_adjacency_lists_a,
        *_spec_adjacency_lists_b,
    ]

    def as_list(self) -> List[tf.Tensor]:
        return [
            typing.cast(tf.Tensor, tf.convert_to_tensor(self.graphs_in_batch)),
            self.node_labels_unique_a,
            self.node_labels_index_a,
            self.node_to_graph_a,
            self.node_labels_unique_b,
            self.node_labels_index_b,
            self.node_to_graph_b,
            *self.adjacency_lists_a,
            *self.adjacency_lists_b,
        ]

    @classmethod
    def get_edges_count(cls) -> int:
        assert len(cls._spec_adjacency_lists_a) == len(cls._spec_adjacency_lists_b)
        return len(cls._spec_adjacency_lists_a)

    @classmethod
    def build_batch_tensors(
            cls,
            batch_samples: List[Dict]) -> 'SourceGraphDiffInput':
        adjacency_lists_a, node_graph_index_a, node_labels_a = merge_samples_with_graph_node_offset(
            cls,
            batch_samples,
            "afterGraph")
        adjacency_lists_b, node_graph_index_b, node_labels_b = merge_samples_with_graph_node_offset(
            cls,
            batch_samples,
            "afterGraph")

        node_labels_unique_a, node_labels_index_a = cls._node_labels_to_unique_indexed(
            SourceGraphDiffInput._string2tensor, node_labels_a)

        node_labels_unique_b, node_labels_index_b = cls._node_labels_to_unique_indexed(
            SourceGraphDiffInput._string2tensor, node_labels_b)

        return SourceGraphDiffInput(
            graphs_in_batch=tf.convert_to_tensor(len(batch_samples), dtype=cls._spec_graphs_in_batch.dtype),
            node_labels_unique_a=node_labels_unique_a,
            node_labels_index_a=node_labels_index_a,
            adjacency_lists_a=tuple(
                typing.cast(tf.Tensor, tf.concat(adjacency_list, 0)) for adjacency_list in adjacency_lists_a.values()),
            node_to_graph_a=tf.convert_to_tensor(node_graph_index_a, dtype=cls._spec_node_to_graph_a.dtype),
            node_labels_unique_b=node_labels_unique_b,
            node_labels_index_b=node_labels_index_b,
            adjacency_lists_b=tuple(
                typing.cast(tf.Tensor, tf.concat(adjacency_list, 0)) for adjacency_list in adjacency_lists_b.values()),
            node_to_graph_b=tf.convert_to_tensor(node_graph_index_b, dtype=cls._spec_node_to_graph_b.dtype),
        )

    @classmethod
    def get_specs(cls) -> List[tf.TensorSpec]:
        return cls._spec

    @classmethod
    def get_shapes(cls) -> List[tf.TensorShape]:
        return [spec.shape for spec in cls.get_specs()]

    @classmethod
    def get_types(cls) -> List[tf.dtypes.DType]:
        return [spec.dtype for spec in cls.get_specs()]

    @classmethod
    def get_adjacency_list_type(cls) -> tf.dtypes.DType:
        return cls._adjacency_type
