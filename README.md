# Graph Neural Networks

October 6, 2023

## 1 Introduction

In this homework assignment, you will explore various aspects of graph-based machine learning using the MUTAG dataset. The MUTAG dataset is a widely used graph classification dataset in the field of cheminformatics. It is designed for predicting the mutagenicity of chemical compounds, making it a crucial resource for toxicology and drug discovery research.

### 1.1 About the MUTAG Dataset

The MUTAG dataset consists of a collection of chemical compounds, where each compound is represented as a graph. In these graphs:

- Nodes represent atoms within the chemical compound, their features indicate the atom type.
- Edges represent chemical bonds between atoms, their features indicate the chemical bond type.
- The dataset is labeled, with each graph labeled as either mutagenic (positive class) or non-mutagenic (negative class).

The goal of this assignment is to develop and evaluate graph convolutional network models for the graph classification task. By predicting whether a chemical compound is mutagenic or non-mutagenic, you’ll gain valuable insights into the applications of graph-based machine learning in cheminformatics.

### 1.2 Dataset Download Instructions

You may access the MUTAG dataset from the Hugging Face Datasets library via the following link:

https://huggingface.co/datasets/graphs-datasets/MUTAG

The dataset comprises a total of 188 small graphs, each representing a molecule.

Please note that the website provides only the training partition, which we will treat as the entire dataset. Additionally, it’s important to highlight that the dataloader provided on the website is non-functional. Consequently, you will need to create your own custom dataloader for this dataset.

### 1.3 Assignment Structure

This assignment is divided into three parts, each focusing on different aspects of graph convolutional networks and network design. You will explore various graph convolution layers, customize your network architecture, tune hyperparameters, and investigate the impact of incorporating both node and edge features.

Let’s begin by implementing different graph convolution layers in Part 1 of the assignment.

## Part 1: Implementing Different Graph Convolution and Pooling Layers

In this part, you will implement different graph convolutional and Pooling layers.

1. **Normal Convolution (Graph Convolution)**

Implement a Normal Convolution (Graph Convolution) layer for your GCN model. The Normal Convolution aggregates neighbor data without any specific attention mechanism. The equation for a Normal Convolution at layer $l$ can be defined as follows:

$$
h_v^{(l+1)} = \sigma\left( W_l \left( \sum_{u \in \mathcal{N}(v)} \frac{h_u^{(l)}}{|\mathcal{N}(v)|} \right) + B_l h_v^{(l)} \right)
$$

where $h_v^{(l)}$ is the embedding of node $v$ at layer $l$, $\mathcal{N}(v)$ is the set of neighbors of node $v$, $\sigma$ is an activation function, and $W_l$ and $B_l$ are the weight matrices for layer $l$.

2. **GraphSAGE (Customized Aggregation)**

Implement a GraphSAGE layer with a customized aggregation function for your GCN model, which could for example be a mean or LSTM aggregator. The equation for GraphSAGE at layer $l$ with an arbitrary aggregator $\text{AGG}(\cdot)$ can be defined as follows:

$$
h_v^{(l+1)} = \sigma\left( W_l \cdot \text{CONCAT}\left( h_v^{(l)}, \text{AGG}(\{ h_u^{(l)}, \forall u \in \mathcal{N}(v) \}) \right) \right)
$$

Note: as you have seen in the lectures, we can use the adjacency matrix to write the above layer more easily. Therefore, feel free to transform the edge indices into an adjacency matrix if you need to.

3. **Attention-based Convolution**

Implement an Attention-based Convolution layer for your GCN model. Attention-based Convolution applies attention mechanisms to neighbor aggregation. The equation for Attention-based Convolution at layer $l$ can be defined as follows:

$$
h_v'^{(l)} = W_l h_v^{(l)}
$$

$$
h_v^{(l+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} a_{vu}^{(l)} \cdot h_u'^{(l)} \right)
$$

The attention score $e_{vu}$ between node $v$ and its neighbor $u$ is calculated as the dot product between a learnable weight vector and the concatenation of their features and passed through a LeakyReLU activation:

$$
e_{vu} = \text{LeakyReLU}( S^\top \cdot \text{CONCAT}( h_v'^{(l)}, h_u'^{(l)} ) )
$$

where $S$ is a learnable weight vector, shared amongst all nodes.

After calculating the attention scores, we apply the softmax function to obtain normalized attention weights:

$$
a_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v) \cup \{v\}} \exp(e_{vk})}
$$

4. **Mean Pooling**

Given a graph with node features $X \in \mathbb{R}^{N \times D}$, where $N$ is the number of nodes and $D$ is the feature dimension, mean pooling computes the graph-level representation $h_{\text{global}}$ as:

$$
h_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} X_i
$$

5. **Max Pooling**

Max pooling selects the maximum value from each feature dimension:

$$
h_{\text{global}}[d] = \max_{i=1}^{N} X_i[d]
$$

As a side note, it’s worth mentioning that there are more advanced graph pooling strategies. DiffPool and SAGPool are notable examples of hierarchical pooling techniques that offer advanced methods for graph coarsening and pooling.

## Part 2: Custom Network Design with Node Features

In this part, you will design a custom neural network architecture that incorporates different types of graph convolutional layers (from Part 1). The steps include:

1. **Custom Network Architecture**: Implement a neural network with user-defined choices for the types and number of graph convolutional layers. For simplicity, the network’s prediction head should only have one pooling layer (Mean or Max) and one Fully Connected Layer.

2. **Data Partitioning**: Split the dataset into training (70%), validation (15%), and test sets (15%) for consistent evaluation.

3. **Hyperparameter Tuning**: Experiment with different hyperparameters such as the number of layers, step size, and the type of convolutional layers (e.g., Normal, GraphSAGE, Attention-based). Train and evaluate multiple network configurations on the training and validation sets.

4. **Performance Evaluation**: Report the results of your experiments, comparing the performance of different network configurations on the validation set. Analyze which combination of hyperparameters works best and provide reasons for your choice. Report your final performances on the test set.

## Part 3: Incorporating Edge Features

In this part, you will extend your custom network to incorporate both node and edge features and compare the results with Part 2.

5. Explain your strategy for incorporating edge features.

6. Modify your GNN models to utilize edge features in addition to node features for the node task.

7. Evaluate the performance of your models and compare the results with those from Part 2 to determine if using edge features improves performance.

## References

[1] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. arXiv preprint arXiv:1710.10903.

[2] Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). Hierarchical graph representation learning with differentiable pooling. In Advances in neural information processing systems.

[3] Lee, J., Lee, I., & Kang, J. (2019). Self-attention graph pooling. In International Conference on Learning Representations (ICLR).
