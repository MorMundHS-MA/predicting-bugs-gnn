from typing import Tuple, Union, List

import tensorflow as tf
from tf2_gnn import GNN, GNNInput
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput

from models.source_graph.node_label_charcnn_embedding_layer import NodeLabelCharCNNEmbeddingLayer, LabelToNodeRepresentationInput
from models.source_graph.diff_graphs.diff_graph_input import SourceGraphDiffInput


class DiffGraphModel(tf.keras.Model):
    def get_config(self):
        super(DiffGraphModel, self).get_config()

    def __init__(
            self,
            graph_node_label_representation_size=64,
            graph_layers_count=6,
            node_label_embedding_size=32,
            final_graph_representation_size=32,
    ):
        super(DiffGraphModel, self).__init__()
        self._node_embeddings = NodeLabelCharCNNEmbeddingLayer(graph_node_label_representation_size)

        gnn_params = GNN.get_default_hyperparameters()
        gnn_params["num_layers"] = graph_layers_count
        gnn_params["message_calculation_class"] = "GGNN"
        gnn_params["hidden_dim"] = node_label_embedding_size
        self._gnn = GNN(
            gnn_params
        )
        # Need to 'build' the GNN layer first with the appropriate tensor shapes
        # noinspection PyTypeChecker
        gnn_shapes = GNNInput(
            node_features=tf.TensorShape((None, graph_node_label_representation_size)),
            adjacency_lists=tuple([tf.TensorShape((None, 2))] * SourceGraphDiffInput.get_edges_count()),
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
            graph_representation_size=final_graph_representation_size,
            num_heads=1,
            weighting_fun="sigmoid"
        )
        self._regression_layer.build(regression_shapes)

        self._compare_dense_output = tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU())
        self._compare = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)

    def call(self, inputs_tuple: Tuple[Union[tf.Tensor, tf.TensorShape], ...], training=None, mask=None) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        graphs_in_batch, \
            node_labels_unique_a, node_labels_index_a, node_to_graph_a, \
            node_labels_unique_b, node_labels_index_b, node_to_graph_b, *adjacency_lists = inputs_tuple

        adjacency_lists_a: List[tf.Tensor] = adjacency_lists[:SourceGraphDiffInput.get_edges_count()]
        adjacency_lists_b: List[tf.Tensor] = adjacency_lists[SourceGraphDiffInput.get_edges_count():]

        label_input_a = LabelToNodeRepresentationInput(node_labels_unique_a, node_labels_index_a)
        label_input_b = LabelToNodeRepresentationInput(node_labels_unique_b, node_labels_index_b)

        node_representation_a = self._node_embeddings(label_input_a)
        node_representation_b = self._node_embeddings(label_input_b)

        num_graphs = tf.convert_to_tensor(graphs_in_batch)

        gnn_input_a = GNNInput(
            node_features=node_representation_a,
            adjacency_lists=tuple(adjacency_lists_a),
            node_to_graph_map=node_to_graph_a,
            num_graphs=num_graphs,
        )

        gnn_input_b = GNNInput(
            node_features=node_representation_b,
            adjacency_lists=tuple(adjacency_lists_b),
            node_to_graph_map=node_to_graph_b,
            num_graphs=num_graphs,
        )

        nodes_a: tf.Tensor = self._gnn(
            inputs=gnn_input_a,
            training=training,
        )

        nodes_b: tf.Tensor = self._gnn(
            inputs=gnn_input_b,
            training=training,
        )

        graph_representation_input_a = NodesToGraphRepresentationInput(
            node_embeddings=nodes_a,
            node_to_graph_map=node_to_graph_a,
            num_graphs=num_graphs,
        )

        graph_representation_input_b = NodesToGraphRepresentationInput(
            node_embeddings=nodes_b,
            node_to_graph_map=node_to_graph_b,
            num_graphs=num_graphs,
        )

        graph_representation_a: tf.Tensor = self._regression_layer(
            inputs=graph_representation_input_a,
            training=training,
        )

        graph_representation_b: tf.Tensor = self._regression_layer(
            inputs=graph_representation_input_b,
            training=training,
        )

        graph_representations = tf.stack([graph_representation_a, graph_representation_b], axis=1)

        # Randomise before/after graph representation
        rnd_index = tf.random.uniform([num_graphs], maxval=2, dtype=tf.int32)
        inverse_index = 1 - rnd_index
        graph_representation_one = tf.gather(graph_representations, rnd_index, axis=1, batch_dims=1)
        graph_representation_two = tf.gather(graph_representations, inverse_index, axis=1, batch_dims=1)

        dense_out_a = self._compare_dense_output(graph_representation_one)
        dense_out_b = self._compare_dense_output(graph_representation_two)

        compare_in = tf.concat([dense_out_a, dense_out_b], axis=1)

        comparison_out = self._compare(compare_in)

        # Comparison is a [-1, 1] scalar, normalize it to [0, 1]
        diff_out = (comparison_out + 1) / 2
        return diff_out, rnd_index

    def train_step(self, data):
        with tf.GradientTape() as tape:
            order_prediction, index = self(data, training=True)
            loss = self.compiled_loss(index, order_prediction)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(index, order_prediction)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
