import argparse
import time
import os
import csv

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, load_data_with_sampling, make_dataset
from mscn.model import SetConv

def predict_dataset(model, data_loader, cuda):
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
        tables, predicates, joins, targets = Variable(tables), Variable(predicates), Variable(joins), Variable(
            targets)
        table_masks, predicate_masks, join_masks = Variable(table_masks), Variable(predicate_masks), Variable(
            join_masks)

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
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

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

        if (num_materialized_samples > 0):
            dicts, column_min_max_vals, min_val, max_val, _, labels_test, _, _, _, test_data = get_train_datasets(num_queries, num_materialized_samples)
            table2vec, column2vec, op2vec, join2vec = dicts

            joins, predicates, tables, samples, label = load_data_with_sampling(file_name, num_materialized_samples)

            samples_test = encode_samples(tables, samples, table2vec)
            predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
            labels_test, _, _ = normalize_labels(label, min_val, max_val)

            print("Number of test samples: {}".format(len(labels_test)))

            max_num_predicates = max([len(p) for p in predicates_test])
            max_num_joins = max([len(j) for j in joins_test])

            # Get test set predictions
            test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
        else:
            joins, predicates, tables, label = load_data(file_name)

            # Get column name dict
            column_names = get_all_column_names(predicates)
            column2vec, idx2column = get_set_encoding(column_names)

            # Get table name dict
            table_names = get_all_table_names(tables)
            table2vec, idx2table = get_set_encoding(table_names)

            # Get operator name dict
            operators = get_all_operators(predicates)
            op2vec, idx2op = get_set_encoding(operators)

            # Get join name dict
            join_set = get_all_joins(joins)
            join2vec, idx2join = get_set_encoding(join_set)

            # Get min and max values for each column
            with open(file_name_column_min_max_vals, 'r', newline='') as f:
                data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
                column_min_max_vals = {}
                for i, row in enumerate(data_raw):
                    if i == 0:
                        continue
                    column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

            _, min_val, max_val = normalize_labels(label)

            # Get feature encoding and proper normalization
            tables_test = encode_tables(tables, table2vec)
            predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
            labels_test, _, _ = normalize_labels(label, min_val, max_val)

            print("Number of test samples: {}".format(len(labels_test)))

            max_num_predicates = max([len(p) for p in predicates_test])
            max_num_joins = max([len(j) for j in joins_test])

            tables_test = pad_data(tables_test, 6)
            predicates_test = pad_data(predicates_test, 13)
            joins_test = pad_data(joins_test, 6)

            # Get test set predictions
            test_data = make_dataset(tables_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)

        test_data_loader = DataLoader(test_data, batch_size=batch_size)
        preds_test, t_total = predict_dataset(model, test_data_loader, cuda)
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

def predict_query(num_materialized_samples):
    """
    Run inference on a single query.
    Returns the cardinality estimate for the query.
    """
    #TODO: Get inputs from MySQL and run inference.

    if (num_materialized_samples > 0):
        checkpoint_path = '/home/magnus/dev/priv/mysql-server/ml/checkpoints/model_sampling.pth'
    else:
        checkpoint_path = '/home/magnus/dev/priv/mysql-server/ml/checkpoints/model.pth'
    checkpoint = torch.load(checkpoint_path)

    model = SetConv(checkpoint['sample_feats'], checkpoint['predicate_feats'], checkpoint['join_feats'], hid_units=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from {os.path.abspath(checkpoint_path)}")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", help="synthetic, scale, or job-light")
    parser.add_argument("--materialized-samples", help="number of materialized samples (default: 0)", type=int, default=0)
    parser.add_argument("--queries", help="number of queries (default: 0)", type=int, default=0)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    predict_workload(args.workload, args.materialized_samples, args.queries, args.batch, args.hid, args.cuda)

if __name__ == "__main__":
    main()