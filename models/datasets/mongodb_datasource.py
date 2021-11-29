import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional

import tensorflow as tf
from bson import ObjectId

from DBConnector import DBConnector


class MongoDBDataset(ABC):
    def __init__(self, db_name: str, collection_name: str):
        self.__db_name = db_name
        self.__collection_name = collection_name

    class BatchLoader(ABC):
        def __init__(self, collection: DBConnector.Collection):
            self._collection = collection

        @abstractmethod
        def _has_labels(self) -> bool:
            pass

        @abstractmethod
        def _get_model_input_types(self):
            pass

        @abstractmethod
        def _get_model_input_shapes(self):
            pass

        @abstractmethod
        def _load_samples(self, batch_object_ids: List[ObjectId]) -> Tuple[tf.Tensor]:
            pass

        def load_batch(self, raw_sample: tf.Tensor) -> Union[Tuple[tf.Tensor], Tuple[Tuple[tf.Tensor], tf.Tensor]]:
            if self._has_labels():
                return self._load_batch_with_labels(raw_sample)
            else:
                return self._load_batch_without_labels(raw_sample)

        def _load_batch_with_labels(self, raw_sample: tf.Tensor) -> Tuple[Tuple[tf.Tensor], tf.Tensor]:
            types = self._get_model_input_types()
            batch_with_labels = tf.py_function(
                func=self._object_id_parser,
                inp=[raw_sample],
                Tout=types + [tf.float32],
                name="batch_loader_with_labels")
            batch = batch_with_labels[:-1]
            labels = batch_with_labels[-1]
            shapes = self._get_model_input_shapes()
            for idx, tensor in enumerate(batch):
                tensor.set_shape(shapes[idx])
            return tuple(batch), labels

        def _load_batch_without_labels(self, raw_sample: tf.Tensor) -> Tuple[tf.Tensor]:
            types = self._get_model_input_types()
            batch = tf.py_function(
                func=self._object_id_parser,
                inp=[raw_sample],
                Tout=types,
                name="batch_loader_no_labels")
            shapes = self._get_model_input_shapes()
            for idx, tensor in enumerate(batch):
                tensor.set_shape(shapes[idx])
            return tuple(batch)

        def get_ids_as_string(self) -> List[Tuple[str, int]]:
            return self._collection.get_ids_as_string()

        def _object_id_parser(self, batch_id_tensors: tf.Tensor) -> Tuple[tf.Tensor]:
            batch_id_strings = [a.decode("ascii") for a in batch_id_tensors.numpy()]
            batch_obj_ids = [ObjectId(batch_id) for batch_id in batch_id_strings if batch_id[0] != "x"]
            return self._load_samples(batch_obj_ids)

    @abstractmethod
    def get_max_nodes_per_batch(self) -> int:
        pass

    @abstractmethod
    def _get_max_node_props(self) -> List[str]:
        pass

    @abstractmethod
    def _inner_get_loader(self, collection: DBConnector.Collection) -> BatchLoader:
        pass

    def get_loader(self, db_connector: DBConnector) -> BatchLoader:
        collection = db_connector.get_collection(
            db_name=self._get_db_name(),
            collection_name=self._get_collection_name(),
            max_node_props=self._get_max_node_props())
        return self._inner_get_loader(collection)

    # noinspection PyMethodMayBeStatic
    def get_max_parallel_calls(self) -> int:
        return tf.data.AUTOTUNE

    def _get_db_name(self) -> str:
        return self.__db_name

    def _get_collection_name(self) -> str:
        return self.__collection_name


class MongoDBDataSource:
    def __init__(
            self,
            db_connector: DBConnector,
            dataset_class: MongoDBDataset,
            dataset_split: Tuple[float, float, float],
            take: Optional[int] = None,
            random_skip: bool = False,
    ):
        assert len(dataset_split) == 3
        assert sum(dataset_split) == 1
        self._db_connector = db_connector
        self._dataset = dataset_class
        self._loader = dataset_class.get_loader(db_connector)
        self._ids: Union[List[Tuple[str, int]], None] = None
        (self._train_split, self._valid_split, self._test_split) = dataset_split
        self._take = take
        self._random_skip = random_skip

    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.get_training_set(), self.get_validation_set(), self.get_test_set()

    def get_training_set(self) -> tf.data.Dataset:
        return self._get_dataset_from_split(0, self._train_split)

    def get_validation_set(self) -> tf.data.Dataset:
        valid_end = self._train_split + self._valid_split
        return self._get_dataset_from_split(self._train_split, valid_end)

    def get_test_set(self) -> tf.data.Dataset:
        valid_end = self._train_split + self._valid_split
        return self._get_dataset_from_split(valid_end, 1)

    def _get_dataset_from_split(self, split_start_percent: float, split_end_percent: float) -> tf.data.Dataset:
        ids = self._get_ids_and_size()
        ids_len = len(ids)
        split_start_index = math.ceil(ids_len * split_start_percent)
        split_end_index = math.ceil(ids_len * split_end_percent)
        ids_in_set = ids[split_start_index: split_end_index]
        return self._dataset_from_ids(ids_in_set)

    def _dataset_from_ids(self, ids: List[Tuple[str, int]]) -> tf.data.Dataset:
        dataset_batches: List[str] = []
        current_batch: List[str] = []
        current_batch_nodes = 0
        max_nodes_per_batch = self._dataset.get_max_nodes_per_batch()
        max_graphs_per_batch = int(max_nodes_per_batch / 1000)
        for sample_id, nodes_in_sample in ids:
            if nodes_in_sample > max_nodes_per_batch:
                print(f"Sample exceeds max nodes per batch. Discarding {sample_id}")
            if current_batch_nodes + nodes_in_sample > max_nodes_per_batch \
                    or len(current_batch) >= max_graphs_per_batch:
                dataset_batches.extend(self._pad_id_batch(current_batch, max_graphs_per_batch))
                current_batch = []
                current_batch_nodes = 0
            current_batch.append(sample_id)
            current_batch_nodes += nodes_in_sample

        dataset_batches.extend(self._pad_id_batch(current_batch, max_graphs_per_batch))

        dataset = tf.data.Dataset \
            .from_tensor_slices(dataset_batches)

        if self._random_skip:
            batches = math.floor(len(dataset_batches) / max_graphs_per_batch)
            dataset = dataset.skip(random.randint(0, batches - 1) * max_graphs_per_batch)

        if self._take is not None:
            dataset = dataset.take(self._take)

        return dataset \
            .cache() \
            .batch(max_graphs_per_batch) \
            .map(map_func=self._loader.load_batch, num_parallel_calls=self._dataset.get_max_parallel_calls())

    def _get_ids_and_size(self) -> List[Tuple[str, int]]:
        if self._ids is None:
            self._ids = self._loader.get_ids_as_string()

        return self._ids

    @staticmethod
    def _pad_id_batch(batch_ids: List[str], ids_in_batch: int):
        object_id_length = 24
        batch_ids += ["x" * object_id_length] * (ids_in_batch - len(batch_ids))
        return batch_ids
