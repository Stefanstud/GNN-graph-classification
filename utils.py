import networkx as nx
from pyvis.network import Network
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_graph_with_pyvis_node_features(dataset, graph_index, name):
    graph_data = dataset[graph_index]
    node_features = graph_data["node_feat"]
    edge_index = graph_data["edge_index"]

    net = Network(notebook=True, width="100%", height="50vw", cdn_resources="in_line")

    for node_id, node_feat in enumerate(node_features):
        title = f"Node {node_id}<br>Features: {node_feat}"
        color = get_color_from_node_feature(node_feat)
        net.add_node(node_id, title=title, color=color)

    for source, target in zip(edge_index[0], edge_index[1]):
        net.add_edge(source, target)

    net.save_graph(name)


def visualize_graph_with_pyvis_edge_features(dataset, graph_index, name):
    graph_data = dataset[graph_index]
    node_features = graph_data["node_feat"]
    edge_index = graph_data["edge_index"]
    edge_features = graph_data["edge_attr"]

    net = Network(notebook=True, width="100%", height="50vw", cdn_resources="in_line")

    for node_id, node_feat in enumerate(node_features):
        title = f"Node {node_id}<br>Features: {node_feat}"
        color = get_color_from_node_feature(node_feat)
        net.add_node(node_id, title=title, color=color)

    for edge_id, (source, target) in enumerate(zip(edge_index[0], edge_index[1])):
        title = f"Features: {edge_features[edge_id]}"
        color = get_color_from_edge_feature(edge_features[edge_id])
        net.add_edge(source, target, color=color, title=title)

    net.save_graph("edge_" + name)


def get_color_from_node_feature(feature_vector):
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#000000",
    ]

    active_feature_index = feature_vector.index(1)

    return colors[active_feature_index]


def get_color_from_edge_feature(edge_vector):
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
    ]

    active_feature_index = edge_vector.index(1)

    return colors[active_feature_index]


def aggregate_edge_features_by_node(x, edge_index, edge_attr):
    edge_index = edge_index.squeeze(0)
    edge_attr = edge_attr.squeeze(0)

    num_nodes = x.size(0)
    num_features = edge_attr.size(-1)

    aggregated_attributes = torch.zeros(num_nodes, num_features).to(x.device)

    # iterate through each edge and sum up the edge attributes for the corresponding source node
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()  #
        aggregated_attributes[src] += edge_attr[i]

    return aggregated_attributes


def aggregate_edge_attr(x, edge_index, edge_attr):
    edge_index = edge_index[0]

    aggregated_attr = torch.full(
        (x.size(0), edge_attr.size(1)), float("-inf"), device=x.device
    )

    for idx, (src, tgt) in enumerate(edge_index.t()):
        aggregated_attr[tgt] = torch.max(aggregated_attr[tgt], edge_attr[idx])

    aggregated_attr[aggregated_attr == float("-inf")] = 0

    return aggregated_attr


def plot_confusion_matrix(cm, title):
    muted_blues = sns.light_palette("blue", as_cmap=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap=muted_blues,
        cbar=False,
        xticklabels=["NOT MUTAGENIC", "MUTAGENIC"],
        yticklabels=["NOT MUTAGENIC", "MUTAGENIC"],
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()


def find_threshold(model, val_loader, edges=False):
    model.eval()
    true_labels = []
    probs_list = []

    for batch in val_loader:
        x, edge_index, edge_attr, labels, adj_matrix, adj_matrix_edges = batch
        x = x[0]
        labels = labels[0].float()

        if edges:
            adj_matrix = adj_matrix_edges[0]
        else:
            adj_matrix = adj_matrix[0]  # Use the node adjacency matrix

        with torch.no_grad():
            outputs = model(x, adj_matrix)
        probs = torch.sigmoid(outputs)
        true_labels.extend(labels.cpu().numpy())
        probs_list.extend(probs.cpu().numpy())

    thresholds = np.arange(0.1, 1, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        predictions = (probs_list > threshold).astype(float)
        f1 = f1_score(true_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
