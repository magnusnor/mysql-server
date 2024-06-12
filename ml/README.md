# Learned Cardinality Estimation

This module contains code for learned cardinality estimation in MySQL, and is a part of the repository `mysql-server`.

## Multi-Set Convolutional Network

The model used for learned cardinality estimation is the `Multi-Set Convolutional Network (MSCN)` model by [Kipf et al.](https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf), and is based on the [repository](https://github.com/andreaskipf/learnedcardinalities) by the same authors.

## Overview

### Directories

A quick overview of the directories in this module:

- `checkpoints`: The model checkpoints created during training.
- `data`: The training data for the MSCN model.
- `dataframes`: The model dataframes created after model training.
- `mscn`: The source code for the MSCN model.
- `results`: The results gathered from running model inference on given workloads.
- `workloads`: The different workloads used for model testing.

### Files:

A quick overview of the files in this module:

- `logging_config.py`: Logging module used for debugging.
- `requirements.txt`: File containing the required Python packages/modules.
- `predict.py`: Runs the model inference.
- `server.py`: Starts the UNIX server needed to communicate with MySQL.
- `train.py`: Trains the MSCN model.
- `visualize.py`: Visualizes the results.

#### Environment File

A `.env` file is required for the MSCN model to predict cardinality estimates for sub-plans, as well as visualizing the results.

Two environment variables are needed:

- `BENCHMARK_RESULTS_PATH`: The absolute path to repository containing the repository for conducting the timing experiments.
- `MODEL_CHECKPOINTS_PATH`: The absolute path to the `checkpoints` folder, as this is needed for the cardinality estimation of sub-plans.

## Quick-Start

### Installation & Setup

Install the necessary Python packages with a package manager e.g., `pip`. Preferably inside of a virtual environment.

```sh
pip install -r requirements.txt
```

Create a `.env` file with the necessary environment variables:

```sh
touch .env
```

```sh
BENCHMARK_RESULTS_PATH=/path/to/benchmark/repository
MODEL_CHECKPOINTS_PATH=/path/to/mysql-server/ml/checkpoints
```

### Training

The MSCN model in this repository comes pre-trained, see `checkpoints`. However, to train the model one can run the following command:

```sh
python train.py --epochs=100
```

This trains the MSCN model for 100 epochs with the default hyperparameters.

Other hyperparameters which can be configured:

- `--queries`: Number of training queries (default: `100000`)
- `--materialized-samples`: Number of materialized base table samples (default: `0`)
- `--batch`: The batch size (default: `1024`)
- `--hid`: The number of hidden units (default `256`)
  
To train the model with CUDA, add the flag:

- `--cuda`

To save the best model during training, add the flag:

- `--save-best-model`

### Prediction

Running model prediction (inference) on a workload can be done using the following command:

```sh
python predict.py <workload>
```

The name of the workload corresponds to the name of the corresponding `.csv` file in the `workloads` folder.

For example, to run model inference and predict cardinality estimates for the `JOB-light` workload:

```sh
python predict.py job-light
```

### Learned Cardinality Estimation in MySQL

Start the MySQL Test Framework Client:

```
cd build/mysql-test
./mtr --start-dirty
```

This starts the UNIX server located in `server.py`, and establishes a connection between MySQL and the MSCN model.

To enable learned cardinality estimation for a given query, set the following optimizer hint active and execute the query:

```sql
SELECT /*+ML_CARDINALITY_ESTIMATION(1) */ FROM t1, t2 WHERE t1.x > t2.y;
```
