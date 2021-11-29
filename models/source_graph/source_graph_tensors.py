from typing import Tuple, Dict, List, Type

import tensorflow as tf

from models.source_graph.source_graph_input import SourceGraphInput


def merge_samples_with_graph_node_offset(input_type: Type[SourceGraphInput], batch_samples, graph_field_name: str) \
        -> Tuple[Dict[str, List[tf.Tensor]], List[int], List[Tuple[int, str]]]:
    node_labels: List[Tuple[int, str]] = []
    adjacency_lists: Dict[str, List[tf.Tensor], ...] = {}
    node_graph_index: List[int] = []
    current_graph_node_offset = 0
    current_graph_index = 0
    for sample in batch_samples:
        nodes_in_current_graph = len(sample[graph_field_name]["nodeLabels"].items())

        node_labels.extend((int(index) + current_graph_node_offset, label) for index, label in
                           sample[graph_field_name]["nodeLabels"].items())

        for edge_type, edge in sample[graph_field_name]['edges'].items():
            if len(edge) == 0:
                adjacency = tf.zeros(shape=(0, 2), dtype=input_type.get_adjacency_list_type())
            else:
                adjacency = tf.convert_to_tensor(edge, dtype=input_type.get_adjacency_list_type())
                adjacency += current_graph_node_offset
            try:
                adjacency_lists[edge_type].append(adjacency)
            except KeyError:
                adjacency_lists[edge_type] = [adjacency]

        node_graph_index.extend([current_graph_index] * nodes_in_current_graph)
        current_graph_index += 1
        current_graph_node_offset += nodes_in_current_graph
    return adjacency_lists, node_graph_index, node_labels
