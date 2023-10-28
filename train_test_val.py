from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch import nn
import torch
import numpy as np
from utils import aggregate_edge_features_by_node


def train(
    model, train_loader, optimizer, criterion=nn.BCEWithLogitsLoss(), threshold=0.5
):
    model.train()
    train_loss = 0.0

    true_labels, preds = (
        [],
        [],
    )

    for batch in train_loader:
        x, edge_index, edge_attr, labels, adj_matrix, _ = batch
        x = x[0]
        labels = labels[0].float()
        adj_matrix = adj_matrix[0]
        outputs = model(x, adj_matrix)
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        probs = torch.sigmoid(outputs)
        predictions = (probs > threshold).float()

        true_labels.extend(labels.cpu().numpy())
        preds.extend(predictions.cpu().numpy())

    train_accuracy = accuracy_score(true_labels, preds)
    train_f1 = f1_score(true_labels, preds)
    return train_loss / len(train_loader), train_accuracy, train_f1


def evaluate(model, val_loader, criterion=nn.BCEWithLogitsLoss(), threshold=0.5):
    model.eval()
    val_loss = 0.0
    val_labels, val_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, edge_index, edge_attr, labels, adj_matrix, _ = batch

            x = x[0]
            labels = labels[0].float()
            adj_matrix = adj_matrix[0]
            outputs = model(x, adj_matrix)
            outputs = outputs.squeeze(-1)

            val_loss += criterion(outputs, labels).item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predictions.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    return val_loss / len(val_loader), val_accuracy, val_f1


def test(model, test_loader, criterion=nn.BCEWithLogitsLoss(), threshold=0.5):
    model.eval()
    test_loss = 0.0
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, edge_index, edge_attr, labels, adj_matrix, _ = batch
            adj_matrix = adj_matrix[0]
            x = x[0]
            labels = labels[0].float()

            outputs = model(x, adj_matrix)
            outputs = outputs.squeeze(-1)

            test_loss += criterion(outputs, labels).item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predictions.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    conf_matrix = confusion_matrix(test_labels, test_preds)

    return test_loss / len(test_loader), test_accuracy, test_f1, conf_matrix


def train_edges(
    model, train_loader, optimizer, criterion=nn.BCEWithLogitsLoss(), threshold=0.5
):
    model.train()
    train_loss = 0.0

    true_labels, preds = (
        [],
        [],
    )

    for batch in train_loader:
        x, edge_index, edge_attr, labels, adj_matrix, adj_matrix_edges = batch
        x = x[0]
        labels = labels[0].float()
        adj_matrix_edges = adj_matrix_edges[0]
        outputs = model(x, adj_matrix_edges)
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        probs = torch.sigmoid(outputs)
        predictions = (probs > threshold).float()

        true_labels.extend(labels.cpu().numpy())
        preds.extend(predictions.cpu().numpy())

    train_accuracy = accuracy_score(true_labels, preds)
    train_f1 = f1_score(true_labels, preds)
    return train_loss / len(train_loader), train_accuracy, train_f1


def evaluate_edges(model, val_loader, criterion=nn.BCEWithLogitsLoss(), threshold=0.5):
    model.eval()
    val_loss = 0.0
    val_labels, val_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, edge_index, edge_attr, labels, adj_matrix, adj_matrix_edges = batch

            x = x[0]
            labels = labels[0].float()
            adj_matrix_edges = adj_matrix_edges[0]
            outputs = model(x, adj_matrix_edges)
            outputs = outputs.squeeze(-1)

            val_loss += criterion(outputs, labels).item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predictions.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    return val_loss / len(val_loader), val_accuracy, val_f1


def test_edges(model, test_loader, criterion=nn.BCEWithLogitsLoss(), threshold=0.5):
    model.eval()
    test_loss = 0.0
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, edge_index, edge_attr, labels, adj_matrix, adj_matrix_edges = batch
            adj_matrix_edges = adj_matrix_edges[0]
            x = x[0]
            labels = labels[0].float()

            outputs = model(x, adj_matrix_edges)
            outputs = outputs.squeeze(-1)

            test_loss += criterion(outputs, labels).item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predictions.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    conf_matrix = confusion_matrix(test_labels, test_preds)

    return test_loss / len(test_loader), test_accuracy, test_f1, conf_matrix


def test_add_edge_to_nodes(
    model, test_loader, criterion=nn.BCEWithLogitsLoss(), threshold=0.5
):
    model.eval()
    test_loss = 0.0
    test_labels, test_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, edge_index, edge_attr, labels, adj_matrix, _ = batch

            x = x[0]
            labels = labels[0].float()
            adj_matrix = adj_matrix[0]
            # aggregate the edge attributes
            aggregated_attributes = aggregate_edge_features_by_node(
                x, edge_index, edge_attr
            )

            # Augment or replace node features with aggregated attributes
            # example where we concatenate them:
            x_combined = torch.cat([x, aggregated_attributes], dim=1)
            outputs = model(x_combined, adj_matrix)
            outputs = outputs.squeeze(-1)

            test_loss += criterion(outputs, labels).item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predictions.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    conf_matrix = confusion_matrix(test_labels, test_preds)

    return test_loss / len(test_loader), test_accuracy, test_f1, conf_matrix
