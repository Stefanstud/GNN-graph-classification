# Graph Neural Networks

October 6, 2023

## Introduction

The MUTAG dataset is a widely used graph classification dataset in the field of cheminformatics. It is designed for predicting the mutagenicity of chemical compounds, making it a crucial resource for toxicology and drug discovery research.

### About the MUTAG Dataset

The MUTAG dataset consists of a collection of chemical compounds, where each compound is represented as a graph. In these graphs:

- Nodes represent atoms within the chemical compound, their features indicate the atom type.
- Edges represent chemical bonds between atoms, their features indicate the chemical bond type.
- The dataset is labeled, with each graph labeled as either mutagenic (positive class) or non-mutagenic (negative class).

The goal of this assignment is to develop and evaluate graph convolutional network models for the graph classification task. By predicting whether a chemical compound is mutagenic or non-mutagenic, you’ll gain valuable insights into the applications of graph-based machine learning in cheminformatics.

## Methods

1. **Graph Convolution**

The Normal Convolution aggregates neighbor data without any specific attention mechanism. The equation for a Normal Convolution at layer $l$ can be defined as follows:

$$
h_v^{(l+1)} = \sigma\left( W_l \left( \sum_{u \in \mathcal{N}(v)} \frac{h_u^{(l)}}{|\mathcal{N}(v)|} \right) + B_l h_v^{(l)} \right)
$$

where $h_v^{(l)}$ is the embedding of node $v$ at layer $l$, $\mathcal{N}(v)$ is the set of neighbors of node $v$, $\sigma$ is an activation function, and $W_l$ and $B_l$ are the weight matrices for layer $l$.

2. **GraphSAGE (Customized Aggregation)**

The equation for GraphSAGE at layer $l$ with an arbitrary aggregator (could also be an LSTM) $\text{AGG}(\cdot)$ can be defined as follows:

$$
h_v^{(l+1)} = \sigma\left( W_l \cdot \text{CONCAT}\left( h_v^{(l)}, \text{AGG}(\{ h_u^{(l)}, \forall u \in \mathcal{N}(v) \}) \right) \right)
$$

Note: we can use the adjacency matrix to write the above layer more easily. 

3. **Attention-based Convolution**

Attention-based Convolution applies attention mechanisms to neighbor aggregation. The equation for Attention-based Convolution at layer $l$ can be defined as follows:

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

## References

[1] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. arXiv preprint arXiv:1710.10903.

[2] Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). Hierarchical graph representation learning with differentiable pooling. In Advances in neural information processing systems.

[3] Lee, J., Lee, I., & Kang, J. (2019). Self-attention graph pooling. In International Conference on Learning Representations (ICLR).
