import argparse
import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import dataset
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, load_data_with_sampling, make_dataset, make_tensors
from mscn.model import SetConv

from dotenv import load_dotenv
load_dotenv()

from logging_config import get_logger

logger = get_logger(__name__)

MODEL_CHECKPOINTS_PATH = os.environ["MODEL_CHECKPOINTS_PATH"]

def run_inference_on_query(model, data_loader, cuda):
    """
    Run inference on a single query.
    Returns the prediction.
    """
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):
        tables, predicates, joins, table_masks, predicate_masks, join_masks = data_batch

        if cuda:
            tables, predicates, joins = tables.cuda(), predicates.cuda(), joins.cuda()
            table_masks, predicate_masks, join_masks = table_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        tables, predicates, joins = Variable(tables), Variable(predicates), Variable(joins)
        table_masks, predicate_masks, join_masks = Variable(table_masks), Variable(predicate_masks), Variable(join_masks)

        t = time.time()

        outputs = model(tables, predicates, joins, table_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total

def run_inference_on_dataset(model, data_loader, cuda):
    """
    Run inference on each query in a given dataset.
    Returns a list of all predictions.
    """
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):
        tables, predicates, joins, targets, table_masks, predicate_masks, join_masks = data_batch

        if cuda:
            tables, predicates, joins, targets = tables.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            table_masks, predicate_masks, join_masks = table_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        tables, predicates, joins, targets = Variable(tables), Variable(predicates), Variable(joins), Variable(targets)
        table_masks, predicate_masks, join_masks = Variable(table_masks), Variable(predicate_masks), Variable(join_masks)

        t = time.time()

        outputs = model(tables, predicates, joins, table_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total

def predict_workload(workload_name, num_materialized_samples, num_queries, batch_size, hid_units, cuda):
    """
    Predicts the cardinality for each query in a given workload.
    Writes the results to a CSV file.
    """

    print(f"Number of materialized base table samples: {num_materialized_samples}")

    if (num_materialized_samples > 0):
        checkpoint_path = 'checkpoints/model_sampling.pth'
    else:
        checkpoint_path = 'checkpoints/model.pth'

    if (os.path.exists(checkpoint_path)):
        checkpoint = torch.load(checkpoint_path)

        if (num_materialized_samples > 0):
            model = SetConv(checkpoint['sample_feats'], checkpoint['predicate_feats'], checkpoint['join_feats'], hid_units)
        else:
            model = SetConv(checkpoint['table_feats'], checkpoint['predicate_feats'], checkpoint['join_feats'], hid_units)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Loaded checkpoint from {os.path.abspath(checkpoint_path)}")

        # Load test data
        file_name = "workloads/" + workload_name

        print(f"Workload:\n{print_workload_name(workload_name)}")

        dicts, column_min_max_vals, min_val, max_val, _, labels_test, _, _, _, test_data = get_train_datasets(num_queries, num_materialized_samples)
        table2vec, column2vec, op2vec, join2vec = dicts

        if (num_materialized_samples > 0):
            joins, predicates, tables, samples, label = load_data_with_sampling(file_name, num_materialized_samples)
            samples_test = encode_samples(tables, samples, table2vec)
        else:
            joins, predicates, tables, label = load_data(file_name)
            tables_test = encode_tables(tables, table2vec)
        
        predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
        labels_test, _, _ = normalize_labels(label, min_val, max_val)

        print("Number of test samples: {}".format(len(labels_test)))

        max_num_predicates = max([len(p) for p in predicates_test])
        max_num_joins = max([len(j) for j in joins_test])

        # Get test set predictions
        if (num_materialized_samples > 0):
            test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
        else:
            test_data = make_dataset(tables_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)

        test_data_loader = DataLoader(test_data, batch_size=batch_size)
        preds_test, t_total = run_inference_on_dataset(model, test_data_loader, cuda)
        print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

        # Unnormalize
        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

        # Print metrics
        print("\nQ-Error " + workload_name + ":")
        print_qerror(preds_test_unnorm, label)

        # Write predictions
        if (num_materialized_samples > 0):
            file_name = "results/predictions_" + workload_name + "-sampling.csv"
        else:
            file_name = "results/predictions_" + workload_name + ".csv"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            for i in range(len(preds_test_unnorm)):
                f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")

def predict_query(query, batch_size=256, cuda=False):
    """
    Run inference on a single query.
    Returns the cardinality estimate for the query.
    """

    checkpoint_path = f"{MODEL_CHECKPOINTS_PATH}/model.pth"
    checkpoint = torch.load(checkpoint_path)

    model = SetConv(checkpoint["table_feats"], checkpoint["predicate_feats"], checkpoint["join_feats"], hid_units=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded checkpoint from {os.path.abspath(checkpoint_path)}")

    column_min_max_vals = checkpoint["column_min_max_vals"]
    max_num_joins = checkpoint["max_num_joins"]
    max_num_predicates = checkpoint["max_num_predicates"]
    min_val = checkpoint["min_val"]
    max_val = checkpoint["max_val"]
    table2vec, column2vec, op2vec, join2vec = checkpoint["dicts"]

    joins = [query["joins"]]
    predicates = [query["predicates"]]
    tables = [query["tables"]]

    tables_test = encode_tables(tables, table2vec)    
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)

    table_tensors, predicate_tensors, join_tensors, table_masks, predicate_masks, join_masks = make_tensors(tables_test, predicates_test, joins_test, max_num_joins, max_num_predicates)
    
    query_representation = dataset.TensorDataset(table_tensors, predicate_tensors, join_tensors, table_masks, predicate_masks, join_masks)

    data_loader = DataLoader(query_representation, batch_size=batch_size)

    prediction_norm, t_total = run_inference_on_query(model, data_loader, cuda=cuda)
    cardinality_estimate = unnormalize_labels(prediction_norm, min_val, max_val)[0]

    t_total_ms = t_total*1000
    
    logger.info(f"Inference time (ms): {t_total_ms}")
    logger.info(f"Cardinality estimate: {cardinality_estimate}")

    return cardinality_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", help="synthetic, scale, or job-light")
    parser.add_argument("--materialized-samples", help="number of materialized samples (default: 0)", type=int, default=0)
    parser.add_argument("--queries", help="number of training queries (default: 100000)", type=int, default=100000)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    predict_workload(args.workload, args.materialized_samples, args.queries, args.batch, args.hid, args.cuda)

if __name__ == "__main__":
    main()