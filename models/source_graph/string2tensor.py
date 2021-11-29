from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf


class String2Tensor:
    _instance: 'String2Tensor' = None

    def __init__(self, node_label_max_chars: int, alphabet_string: str):
        self._node_label_max_chars = node_label_max_chars

        # "0" is PAD, "1" is UNK
        self._alphabet_dict: Dict[str, int] = {char: idx + 2 for (idx, char) in enumerate(alphabet_string)}
        self._alphabet_dict["PAD"] = 0
        self._alphabet_dict["UNK"] = 1

    @staticmethod
    def configure_default(node_label_max_chars: int, alphabet_string: str):
        String2Tensor._instance = String2Tensor(node_label_max_chars, alphabet_string)

    @staticmethod
    def get_default() -> 'String2Tensor':
        if String2Tensor._instance is None:
            raise RuntimeError("Default instance not configured")
        return String2Tensor._instance

    @staticmethod
    def char_tensors_to_unique_indexed(string_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        node_label_chars_unique, node_label_chars_indices = np.unique(string_tensor, axis=0, return_inverse=True)
        return (
            tf.convert_to_tensor(node_label_chars_unique, dtype=tf.uint8),
            tf.convert_to_tensor(node_label_chars_indices, dtype=tf.int32)
        )

    def strings_to_tensor(self, strings: List[Tuple[int, str]]) -> tf.Tensor:
        node_label_chars = np.zeros(shape=(len(strings), self._node_label_max_chars), dtype=np.uint8)

        for node, label in strings:
            for (char_idx, label_char) in enumerate(label[:self._node_label_max_chars].lower()):
                node_label_chars[int(node), char_idx] = self._alphabet_dict.get(label_char, self._alphabet_dict["UNK"])

        return tf.convert_to_tensor(node_label_chars)

    def get_node_label_max_chars(self) -> int:
        return self._node_label_max_chars
