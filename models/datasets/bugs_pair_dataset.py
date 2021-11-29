from typing import List, Tuple

import tensorflow as tf
from bson import ObjectId

from DBConnector import DBConnector
from datasets.mongodb_datasource import MongoDBDataset
from source_graph.diff_graphs.diff_graph_input import SourceGraphDiffInput


class BugFixPairDataset(MongoDBDataset):
    def __init__(
            self,
            max_nodes_per_batch: int):
        self._max_nodes_per_batch = max_nodes_per_batch
        super().__init__(db_name="data", collection_name="diffGraphSamples")

    def get_max_nodes_per_batch(self) -> int:
        return self._max_nodes_per_batch

    def _get_max_node_props(self) -> List[str]:
        return ["afterNodeCount", "beforeNodeCount"]

    def _inner_get_loader(self, collection: DBConnector.Collection) -> MongoDBDataset.BatchLoader:
        return BugFixPairDataset.Loader(collection)

    class Loader(MongoDBDataset.BatchLoader):
        def _has_labels(self) -> bool:
            return False

        def _get_model_input_types(self):
            return SourceGraphDiffInput.get_types()

        def _get_model_input_shapes(self):
            return SourceGraphDiffInput.get_shapes()

        def _load_samples(self, batch_object_ids: List[ObjectId]) -> Tuple[tf.Tensor]:
            batch_samples = self._collection.get_samples(batch_object_ids)
            batch = SourceGraphDiffInput.build_batch_tensors(batch_samples)
            return tuple(batch.as_list())