import argparse
import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets
from mscn.model import SetConv
from predict import predict_dataset

import pandas as pd


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def validate(model, val_data_loader, cuda, min_val, max_val):
    model.eval()
    loss_total = 0

    for data_batch in val_data_loader:
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        
        with torch.no_grad():
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
        
    val_qerror = loss_total / len(val_data_loader)
    return val_qerror


def train(num_queries, num_epochs, num_materialized_samples, batch_size, hid_units, cuda):
    # Load training and validation data
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, _, _, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    if (num_materialized_samples > 0):
        sample_feats = len(table2vec) + num_materialized_samples
    else:
        table_feats = len(table2vec)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    if (num_materialized_samples > 0):
        model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)
    else:
        model = SetConv(table_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # Init data for visualization
    train_qerrors, val_qerrors = [], []

    best_val_qerror = float('inf')
    if (num_materialized_samples > 0):
        model_checkpoint_path = os.path.join('checkpoints', 'model_sampling.pth')
    else:
        model_checkpoint_path = os.path.join('checkpoints', 'model.pth')
    epoch_start = 0

    if (os.path.exists(model_checkpoint_path)):
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        print(f"Loaded checkpoint from {os.path.abspath(model_checkpoint_path)}")

    model.train()
    for epoch in range(epoch_start, epoch_start+num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):
            tables, predicates, joins, targets, table_masks, predicate_masks, join_masks = data_batch

            if cuda:
                tables, predicates, joins, targets = tables.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                table_masks, predicate_masks, join_masks = table_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            tables, predicates, joins, targets = Variable(tables), Variable(predicates), Variable(joins), Variable(
                targets)
            table_masks, predicate_masks, join_masks = Variable(table_masks), Variable(predicate_masks), Variable(
                join_masks)

            optimizer.zero_grad()
            outputs = model(tables, predicates, joins, table_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        # Training Q-Error
        train_qerror = loss_total / len(train_data_loader)
        train_qerrors.append(train_qerror)

        # Validation Q-Error
        val_qerror = validate(model, test_data_loader, cuda, min_val, max_val)
        val_qerrors.append(val_qerror)

        print(f"Epoch {epoch}, Training loss: {train_qerror}, Validation loss: {val_qerror}")

        if (val_qerror < best_val_qerror):
            best_val_qerror = val_qerror

            if not os.path.exists(os.path.dirname(model_checkpoint_path)):
                os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
                print(f"Directory '{os.path.dirname(model_checkpoint_path)}' created at {os.path.abspath(os.path.dirname(model_checkpoint_path))}")
            
            # Save model checkpoint at the current epoch
            if (num_materialized_samples > 0):
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sample_feats': sample_feats,
                'predicate_feats': predicate_feats,
                'join_feats': join_feats,
            }, model_checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'table_feats': table_feats,
                    'predicate_feats': predicate_feats,
                    'join_feats': join_feats,
                }, model_checkpoint_path)

            print(f"New best model saved for epoch {epoch} at {os.path.abspath(model_checkpoint_path)}")
    
    if not os.path.exists("./dataframes"):
        os.makedirs("./dataframes", exist_ok=True)
    
    # Save the model training as a DataFrame
    df = pd.DataFrame({'training_loss': train_qerrors, 'validation_loss': val_qerrors, 'epochs': num_epochs})
    if (num_materialized_samples > 0):
        df.to_pickle("./dataframes/model_training_sampling.pkl")
    else:
        df.to_pickle("./dataframes/model_training.pkl")
    print(f"Model training saved as DataFrame at {os.path.abspath('./dataframes/model_training.pkl')}")

    # Get final training and validation set predictions
    preds_train, t_total = predict_dataset(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict_dataset(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", help="number of training queries (default: 100000)", type=int, default=100000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--materialized-samples", help="number of materialized samples (default: 0)", type=int, default=0)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    train(args.queries, args.epochs, args.materialized_samples, args.batch, args.hid, args.cuda)


if __name__ == "__main__":
    main()
