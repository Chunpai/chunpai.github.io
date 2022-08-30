---
title: "Graph Neural Networks"
layout: post
tags: [graph, deep-learning]
author: Chunpai
---

* TOC
{: toc}

## Node Embedding

Given a graph, we would like to learn the embedding of each node, which maps nodes to d-dimensional embedding such that similar nodes in the graph are embeded close together. That is, given two nodes $u$ and $v$, we would like

$$
similarity(u, v) \approx \mathbf{z}_u^\top\mathbf{z}_v
$$

### Encoder and Decoder

There are two key components to learn the node embeddings: encoder and decoder.
Encoder maps each node to a low-dimensional vector:

$$
ENC(v) = \mathbf{z}_v
$$

The most simple encoding approach is the embedding-lookup, which we call it as shallow encoding.  It has been used in many methods such as DeepWalk and node2vec.

*Limitations of Shallow Encoding*:

1. Each node is assigned a unique embedding vector.
2. No sharing of parameters between nodes.
3. Cannot generate embeddings for nodes that are not seen during training.
4. Do not incorporate node features.

Decoder is just the similarity function that specifies how the relationships in vector space map to the relationships in the origin network:

$$
\mathbf{z}_u^\top\mathbf{z}_v \approx similarity(u, v)
$$

Under this encoder decoder framework, the objective is to maximize the $\mathbf{z}_u^\top\mathbf{z}_v$ for node pairs $(u,v)$ that are similar. But we should define the similarity measure that matches different applications. For example, we could have different notions of node similarity as follows:

1. Naive: similar if two nodes are connected.
2. Neighborhood overlap.
3. Random walk approaches.

### Random Walk Embeddings

The idea of using random walk to learn node embedding is if random walk starting from node $u$ visits node $v$ with high probability, then $u$ and $v$ are similar that contains high order multi-hop information. We could define the similarity function as $\mathbf{z}\_{u}^\top \mathbf{z}_{v} \approx$ probability that $u$ and $v$ co-occur on a random walk over the graph.

Generally speaking, we first estimate probability of visiting node $v$ on a random walk starting from node $u$ using some random walk strategy $R$, denoted by $P_{R}(v\mid u)$. Then we optimize embeddings to encode these random walk statistics such as

$$
\text{cosine}(\mathbf{z}_u, \mathbf{z}_v) \propto P_{R}(v\mid u)
$$

**Advantages**:

1. Expressivity: flexible stochastic definition of node similarity that incorporates both local and higher-order neighborbood information.
2. Efficiency: do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks.

**Objective**

Given a graph $G=(V,E)$, our goal is to learn a mapping function $f_{\theta} :u\rightarrow \mathbb{R}^d$ such that $f_{\theta}(u) = \mathbf{z}\_{u}$. We denote $N_{R}(u)$ the multiset of nodes visited on random walks starting from node $u$ by strategy $R$, which defines the length of random walk, etc.

We could leverage the maximum likelihood estimator to optimize the node embeddings (parameters) to maximize the likelihood of random walk co-occurrences (assuming we use shallow encoding and we only need to optimize $\mathbf{z}_u$ directly rather than over $\theta$)

$$
\max_{f} \sum_{u\in V} \log p(N_R(u) \mid \mathbf{z}_u)
$$

which is equivalent to

$$
\max_{f} \sum_{u\in V}\sum_{v\in N_{R}(u)} \log p(v \mid \mathbf{z}_u)
$$

where we could set

$$
p(v \mid \mathbf{z}_u) = \frac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\top\mathbf{z}_n)}
$$

to make node $v$ to be most similar to node $u$. Therefore, the objective is to minimize the negative log-likelihood:

$$
\min_{f} \sum_{u\in V}\sum_{v\in N_{R}(u)} -\log (\frac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\top\mathbf{z}_n)})
$$

**Negative Sampling**

As we could see in the objective function, it contains a nested sum over all nodes, which makes it computationally infeasible for large graphs. However, we could approximate the normalization term in softmax function [3]. Instead of normalizing w.r.t all nodes, we could just normalize against $k$ random "negative samples" (nodes $n_i \notin N_{R}(u)$).

$$
\log (\frac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\top\mathbf{z}_n)}) \approx \log( \sigma(\mathbf{z}_u^\top\mathbf{z}_v)) - \sum_{i=1}^{k} \log(\sigma(\mathbf{z}_u^\top\mathbf{z}_{n_i}))
$$

where $k$ negative nodes are sampled with probability proportional to their degrees.

Technically, LHS and RHS are two different objectives. But to maximize the log-likelihood with softmax is similar to maximize the "margin" between "positive" nodes and "negative" nodes. Negative sampling is a form of Noise Contrastive Estimation (NEC).

#### Node2Vec

We use short fixed-length and unbiased random walks starting from each node to find $N_{R}(u)$ or define the similarity, which *could* be too constrained.

The main idea of Node2Vec [4] is to use biased 2nd order random walk strategy $R$ to find better $N_R(u)$ of each node $u$ and leads to rich node embeddings.

Node2Vec uses flexible, biased random walks that can trade-off between local and global views of the network. It uses the BFS to have micro-view of neighborhood and the DFS to have macro-view of neighborhood.

|  ![Node2Vec](/assets/img/node2vec.png)  |
| :------------------------------------: |
| Figure 1.$N_R(u)$ with Biased Walks. |

However, no one method wins in all cases, and Node2Vec performs better on node classification while alternative methods perform better on link prediction. For example, when a random walk starts from a low degree node and the walk may get stuck in a dense cluster with high degree nodes and neglect the low-degree neighbors. This may be the reason that Node2Vec performs better on node classification, which allows the tuning between local information and global informatoin aggregation.

Therefore, we must choose definition of node similarity that matches different applications.





#### Relationship to Matrix Factorization

If we define the node similarity on the graph as the connection of two nodes, that is if two nodes are similar if they are connected by an edge, then we could set $\mathbf{z}\_{u}^\top\mathbf{z}\_{v} = A\_{u,v}$ which is the $(u,v)$ entry of the graph adjacency matrix $A$. Therefore, we could set the objective function 

$$
  \min_{Z} \|A - Z^{\top}Z\|_2
$$

Therefore, the inner product decoder with node similarity defined by edge connectvity is equivalent to marix factorization of adjacency matrix $A$.


There is a connection between random walk embedding and matrix factorization as well. For example, DeepWalk is equivalent to matrix factorization of the following complex matrix expression [5]:

$$
  \log\left(vol(G)\left(\frac{1}{T}\sum_{r=1}^{T}(D^{-1}A)^r\right)D^{-1}\right)-\log b
$$



#### Limitations of Node Embedding via Random Walks and Matrix Factorization

We view these node embedding methods as shallow embedding methods, which have following limitations:

1. They cannot obtain embeddings for nodes not in the training set. 
2. They cannot capture structural similarity (inherently "transductive").
3. They cannot utilize node, edge and graph features.



## Graph Neural Networks

We have seen many successful deep neural networks on sequence and grid data. Here we will see why it is difficult to generalize to graph data. 

Assume we have a graph $G=(V, E)$ 
* $V$ is the vertex set.
* $A$ is the adjacency matrix.
* $X\in \mathbb{R}^{m\times|V|}$ is a matrix of node features.
* $v$ is a node in V, and $N(v)$ denotes the set of neighbors of $v$. 


### Why Is It Difficult?

The navie approach is to joint adjacency matrix with node features, and then feed this augmented matrix into a deep neural network such MLP or CNN. However, this approach has 2 main limitations: 

1. It is not applicable to graphs of different size. 
2. It is also sensitive to node ordering since graph does not have a canonical order of the nodes. In other words, graph is permutation invariant. 



### Deep Graph Encoders



### Graph Convolutional Networks

### GraphSAGE

## Reference

[1] [Geometric Deep Learning Course](https://geometricdeeplearning.com/lectures/)

[2] [Machine Learning with Grpahs](http://web.stanford.edu/class/cs224w/)

[3] Goldberg, Yoav, and Omer Levy. "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method." arXiv preprint arXiv:1402.3722 (2014).

[4] Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

[5] Qiu, Jiezhong, et al. "Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec." Proceedings of the eleventh ACM international conference on web search and data mining. 2018.
