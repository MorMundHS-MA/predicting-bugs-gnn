# Predicting source code bugs using graph neural networks

## About

This repository contains the code for training the graph neural network used in my master's thesis *Predicting source code bugs using graph neural networks*. The thesis itself as well as the dataset are available under [Releases](https://github.com/MorMundHS-MA/predicting-bugs-gnn/releases/tag/thesis).

### Note

The model does not train successfully on the problem, only reaching accuracies barely above randomness. For a discussion of this, I recommend reading my thesis. The code and dataset are provided in the context of the thesis for future research.

## Usage

### Installation

This project requires [Poetry](https://python-poetry.org/) and a GPU with CUDA setup for TensorFlow is recommended. The version of TensorFlow used in this project works best with Python 3.7. Upgrading Python or TensorFlow may break things.

Run `poetry install` to install the projects Python dependencies.

### Training

The graph neural network can be trained by running the command `python main.py -m [MONGO_DB_URI]` in the poetry environment.
The program requires a MongoDB instance running at `MONGO_DB_URI`.
Graph samples are expected to be in a collection `bugFixSamples` in a database `data`. The dataset provided in the ['Releases' section](https://github.com/MorMundHS-MA/predicting-bugs-gnn/releases/download/thesis/bug_graph_samples.json.gz) can be imported with the command:

```sh
gunzip -c bug_graph_samples.json.gz | mongoimport --uri=[MONGO_DB_URI] --db data --collection bugFixSamples
```
