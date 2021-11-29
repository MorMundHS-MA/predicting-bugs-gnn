import typing
from typing import List, Tuple, Union, Dict, Callable
import collections

import tensorflow as tf
from tf2_gnn import GNN, GNNInput
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput

from models.source_graph.node_label_charcnn_embedding_layer import NodeLabelCharCNNEmbeddingLayer, LabelToNodeRepresentationInput
from models.source_graph.source_graph_input import SourceGraphInput
from models.source_graph.string2tensor import String2Tensor
from models.source_graph.source_graph_tensors import merge_samples_with_graph_node_offset

graph_node_label_representation_size = 64
graph_layers_count = 6
node_label_embedding_size = 128
language = "java"
program_graph_edge_types = ["Child", "NextToken", "LastUsedVariable", "LastUse", "LastWrite", "LastLexicalUse",
                            "ComputedFrom", "ReturnsTo", "FormalArgName", "GuardedBy",
                            "GuardedByNegation", "BindsToSymbol"]


class SourceGraphRegressionInput(SourceGraphInput):
    def __init__(self, graphs_in_batch: tf.Tensor, node_labels_unique: tf.Tensor,
                 node_labels_index: tf.Tensor, node_to_graph: tf.Tensor, regression_labels: tf.Tensor,
                 adjacency_lists: Tuple[tf.Tensor]):
        self.graphs_in_batch = graphs_in_batch
        self.node_labels_unique = node_labels_unique
        self.node_labels_index = node_labels_index
        self.node_to_graph = node_to_graph
        self.regression_labels = regression_labels
        self.adjacency_lists = adjacency_lists

    _string2tensor = String2Tensor.get_default()
    _spec_graphs_in_batch = tf.TensorSpec(shape=(), dtype=tf.int32)
    _spec_node_labels_unique = tf.TensorSpec(shape=(None, _string2tensor.get_node_label_max_chars()), dtype=tf.uint8)
    _spec_node_labels_index = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    _spec_node_to_graph = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    _spec_regression_labels = tf.TensorSpec(shape=(None,), dtype=tf.float32)
    _spec_adjacency_lists = [tf.TensorSpec((None, 2), dtype=tf.int32)] * len(program_graph_edge_types)

    _spec = [
        _spec_graphs_in_batch,
        _spec_node_labels_unique,
        _spec_node_labels_index,
        _spec_node_to_graph,
        _spec_regression_labels,
        *_spec_adjacency_lists
    ]

    def as_list(self) -> List[tf.Tensor]:
        return [
            typing.cast(tf.Tensor, tf.convert_to_tensor(self.graphs_in_batch)),
            self.node_labels_unique,
            self.node_labels_index,
            self.node_to_graph,
            self.regression_labels,
            *self.adjacency_lists
        ]

    @classmethod
    def build_batch_tensors(
            cls,
            batch_samples: List[Dict],
            get_sample_label: Callable[[Dict], float],
            graph_field_name="contextGraph") -> 'SourceGraphRegressionInput':
        adjacency_lists, node_graph_index, node_labels = merge_samples_with_graph_node_offset(
            cls,
            batch_samples,
            graph_field_name)

        node_labels_unique, node_labels_index = cls._node_labels_to_unique_indexed(
            SourceGraphRegressionInput._string2tensor, node_labels)

        batch_regression_labels = [get_sample_label(sample) for sample in batch_samples]

        return SourceGraphRegressionInput(
            node_labels_unique=node_labels_unique,
            node_labels_index=node_labels_index,
            adjacency_lists=tuple(
                typing.cast(tf.Tensor, tf.concat(adjacency_list, 0)) for adjacency_list in adjacency_lists.values()),
            regression_labels=tf.convert_to_tensor(batch_regression_labels, dtype=cls._spec_regression_labels.dtype),
            node_to_graph=tf.convert_to_tensor(node_graph_index, dtype=cls._spec_node_to_graph.dtype),
            graphs_in_batch=tf.convert_to_tensor(len(batch_samples), dtype=cls._spec_graphs_in_batch.dtype),
        )

    @classmethod
    def get_edges_count(cls) -> int:
        return len(cls._spec_adjacency_lists)

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
        return cls._spec_adjacency_lists[0].dtype


class SourceGraphRegressionModel(tf.keras.Model):
    def get_config(self):
        super(SourceGraphRegressionModel, self).get_config()

    def __init__(self):
        super(SourceGraphRegressionModel, self).__init__()
        self._node_embeddings = NodeLabelCharCNNEmbeddingLayer(graph_node_label_representation_size)

        gnn_params = GNN.get_default_hyperparameters()
        gnn_params["num_layers"] = 6
        gnn_params["message_calculation_class"] = "GGNN"
        gnn_params["hidden_dim"] = node_label_embedding_size
        self._gnn = GNN(
            gnn_params
        )
        # Need to 'build' the GNN layer first with the appropriate tensor shapes
        # noinspection PyTypeChecker
        gnn_shapes = GNNInput(
            node_features=tf.TensorShape((None, graph_node_label_representation_size)),
            adjacency_lists=tuple([tf.TensorShape((None, 2))] * SourceGraphRegressionInput.get_edges_count()),
            node_to_graph_map=tf.TensorShape((None,)),
            num_graphs=tf.TensorShape(1)
        )
        self._gnn.build(gnn_shapes)

        # Need to 'build' the GNN layer first with the appropriate tensor shapes, only node_embeddings is used but lets
        # supply all of the to be nice.
        # noinspection PyTypeChecker
        regression_shapes = NodesToGraphRepresentationInput(
            node_embeddings=tf.TensorShape((None, node_label_embedding_size)),
            node_to_graph_map=tf.TensorShape((None,)),
            num_graphs=tf.TensorShape(1)
        )

        self._regression_layer = WeightedSumGraphRepresentation(
            graph_representation_size=1,
            num_heads=1,
            weighting_fun="sigmoid",
            transformation_mlp_activation_fun="leaky_relu",
            transformation_mlp_layers=[128, 32, 8],
            transformation_mlp_use_biases=True,
        )
        self._regression_layer.build(regression_shapes)

    def call(self, inputs_tuple: Tuple[Union[tf.Tensor, tf.TensorShape], ...], training=None, mask=None) -> tf.Tensor:
        graphs_in_batch, node_labels_unique, \
            node_labels_index, node_to_graph, \
            regression_labels, *adjacency_lists = inputs_tuple

        # Something adds a dimension of size 1 during validation (or it gets remove by something during training)
        unique_labels_as_characters = self._squeeze_excess_dim(node_labels_unique, expected_dims=2)
        node_labels_to_unique_labels = self._squeeze_excess_dim(node_labels_index)
        label_input = LabelToNodeRepresentationInput(unique_labels_as_characters, node_labels_to_unique_labels)

        node_representation = self._node_embeddings(label_input)
        node_to_graph_map = self._squeeze_excess_dim(node_to_graph)
        num_graphs = tf.convert_to_tensor(graphs_in_batch)

        gnn_input = GNNInput(
            node_features=node_representation,
            adjacency_lists=tuple(adjacency_lists),
            node_to_graph_map=node_to_graph_map,
            num_graphs=num_graphs,
        )

        nodes: tf.Tensor = self._gnn(
            inputs=gnn_input,
            training=training,
        )

        graph_representation_input = NodesToGraphRepresentationInput(
            node_embeddings=nodes,
            node_to_graph_map=node_to_graph_map,
            num_graphs=num_graphs,
        )

        graph_representation: tf.Tensor = self._regression_layer(
            inputs=graph_representation_input,
            training=training,
        )

        return graph_representation

    def train_step(self, data):
        batch, labels = data

        with tf.GradientTape() as tape:
            complexity_prediction = self(batch, training=True)
            loss = self.compiled_loss(labels, complexity_prediction)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, complexity_prediction)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _squeeze_excess_dim(tensor: tf.Tensor, expected_dims: int = 1) -> tf.Tensor:
        if len(tensor.shape.dims) > expected_dims:
            return tf.squeeze(tensor, expected_dims)
        else:
            return tensor
