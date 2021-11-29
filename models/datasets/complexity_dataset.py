from typing import List, Tuple
from typing_extensions import Literal

import tensorflow as tf
from bson import ObjectId

from DBConnector import DBConnector
from datasets.mongodb_datasource import MongoDBDataset
from source_graph.regression.source_graph_regression_model import SourceGraphRegressionInput


class ComplexityDataset(MongoDBDataset):
    def __init__(
            self,
            max_nodes_per_batch: int,
            complexity_measure: Literal["cyclomaticComplexity", "cognitiveComplexity"]):
        self._max_nodes_per_batch = max_nodes_per_batch
        self._label = complexity_measure
        super().__init__(db_name="data", collection_name="complexitySamples")

    def get_max_nodes_per_batch(self) -> int:
        return self._max_nodes_per_batch

    def _get_max_node_props(self) -> List[str]:
        return ["nodeLabelCount"]

    def _inner_get_loader(self, collection: DBConnector.Collection) -> MongoDBDataset.BatchLoader:
        return ComplexityDataset.Loader(collection, self._label)

    class Loader(MongoDBDataset.BatchLoader):
        def __init__(self, collection: DBConnector.Collection, label: str):
            self._label = label
            super().__init__(collection)

        def _has_labels(self) -> bool:
            return True

        def _get_model_input_types(self):
            return SourceGraphRegressionInput.get_types()

        def _get_model_input_shapes(self):
            return SourceGraphRegressionInput.get_shapes()

        def _load_samples(self, batch_obj_ids: List[ObjectId]) -> Tuple[tf.Tensor]:
            batch_samples = self._collection.get_samples(batch_obj_ids)
            batch_labels = tf.convert_to_tensor(
                [sample[self._label] for sample in batch_samples],
                dtype=tf.float32)
            batch = SourceGraphRegressionInput.build_batch_tensors(
                batch_samples,
                lambda sample: sample[self._label]).as_list()
            return tuple(batch + [batch_labels])
