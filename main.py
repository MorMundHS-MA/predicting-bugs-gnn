import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from pymongo import MongoClient
from tensorflow.python.keras.callbacks import TensorBoard

from models.DBConnector import DBConnector
from models.source_graph.string2tensor import String2Tensor


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    mongo_uri = args.mongo_uri

    debug_mode = args.debug
    if debug_mode:
        tf.config.run_functions_eagerly(True)

    client = MongoClient(mongo_uri)
    db = DBConnector(client)

    max_nodes_per_batch = args.max_nodes

    String2Tensor.configure_default(
        alphabet_string="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        node_label_max_chars=19
    )

    run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs = f"logs/{run_start_time}"
    from datasets import MongoDBDataSource, BugFixDataset
    from source_graph.binary_classification.source_graph_binary_classification_model \
        import SourceGraphBinaryPredictionModel

    dataset_source = MongoDBDataSource(
        db_connector=db,
        dataset_class=BugFixDataset(max_nodes_per_batch),
        dataset_split=(0.8, 0.1, 0.1)
    )

    (train, valid, test) = dataset_source.get_datasets()

    model = SourceGraphBinaryPredictionModel()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ])

    epochs = 20

    checkpoint_path = f"checkpoints/{run_start_time}/"

    os.mkdir(checkpoint_path)
    checkpoints_path = checkpoint_path + "{epoch:02d}-{val_loss:.2f}.hdf5"

    tboard_callback = TensorBoard(log_dir=logs,
                                  histogram_freq=1,
                                  update_freq="batch")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoints_path,
        monitor="val_loss",
        mode="max",
        save_best_only=True
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        patience=2
    )

    model.fit(
        x=train.prefetch(100).shuffle(100),
        validation_data=valid.prefetch(100),
        epochs=epochs,
        callbacks=[
            tboard_callback,
            checkpoint_callback,
            early_stop_callback
        ]
    )

    result = model.evaluate(test)
    print(f"Test result: {result}")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mongo-uri",
        dest="mongo_uri",
        type=str,
        help="URI of the MongoDB instance which has the graph samples for training.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        default=False,
        type=bool,
        help="Disable autograph to allow debugging of TensorFlow Python code.",
    )
    parser.add_argument(
        "-n",
        "--max-batch-nodes",
        dest="max_nodes",
        required=False,
        default=100_000,
        type=int,
        help="Maximum amount of graph nodes per training batch.",
    )

    return parser


if __name__ == "__main__":
    main()
