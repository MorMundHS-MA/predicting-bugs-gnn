from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf

from models.source_graph.string2tensor import String2Tensor


class SourceGraphInput(ABC):
    @classmethod
    def _node_labels_to_unique_indexed(
            cls, string2tensor: String2Tensor, node_labels: List[Tuple[int, str]]) -> Tuple[tf.Tensor, tf.Tensor]:
        node_label_tensor = string2tensor.strings_to_tensor(node_labels)
        return string2tensor.char_tensors_to_unique_indexed(node_label_tensor)

    @classmethod
    @abstractmethod
    def get_edges_count(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_specs(cls) -> List[tf.TensorSpec]:
        pass

    @classmethod
    @abstractmethod
    def get_shapes(cls) -> List[tf.TensorShape]:
        pass

    @classmethod
    @abstractmethod
    def get_types(cls) -> List[tf.dtypes.DType]:
        pass

    @classmethod
    @abstractmethod
    def get_adjacency_list_type(cls) -> tf.dtypes.DType:
        pass
