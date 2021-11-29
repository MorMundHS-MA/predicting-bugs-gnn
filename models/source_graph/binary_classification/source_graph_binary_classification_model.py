from typing import Tuple, Union

import tensorflow as tf

from models.source_graph.regression.source_graph_regression_model import SourceGraphRegressionModel


class SourceGraphBinaryPredictionModel(tf.keras.Model):
    def __init__(self):
        super(SourceGraphBinaryPredictionModel, self).__init__()
        self._regressionModel = SourceGraphRegressionModel()
        self._output_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def get_config(self):
        super(SourceGraphBinaryPredictionModel, self).get_config()

    def call(self, inputs: Tuple[Union[tf.Tensor, tf.TensorShape], ...], training=None, mask=None):
        regression_output = self._regressionModel(inputs)
        output = self._output_layer(regression_output)
        return output

    def train_step(self, data):
        batch, labels = data

        with tf.GradientTape() as tape:
            prediction = self(batch, training=True)
            loss = self.compiled_loss(labels, prediction)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, prediction)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
