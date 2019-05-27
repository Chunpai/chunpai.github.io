---
layout: post
tags: meta-learning
author: Chunpai
---

This note is based on the Chelsea Finn's dissertation "Learning to Learn with Gradients". 

* TOC
{: toc}
## Problem Statement

The goal of **few shot meta-learning** is to train a model that can quickly adapt to a new task using only a few data points and training iterations, which can be accomplished by training on a set of tasks during a meta-learning phase.  During meta-learning, the model, denoted $f: \mathbb{x} \rightarrow \mathbb{y}$ , is trained to be able to adapt to a large or infinite number of tasks. 



### Notations

Each task $ \mathcal{T}=\left\{\mathcal{L}(\theta, \mathcal{D}), p\left(\mathbf{x}_{1}\right), p\left(\mathbf{x}_{\mathrm{t}+1} \mid \mathbf{x}_{\mathrm{t}}, \mathbf{y}_{\mathrm{t}}\right), \mathrm{H}\right\} $ consists of 

- $\mathcal{L}(\theta, \mathcal{D})$ , a loss function with model's parameter $\theta$ and a dataset $\mathcal{D}$.  In supervised learning problem, $\mathcal{D}=\left\{\left(\mathbf{x}_{1}, \mathbf{y}_{1}\right)^{(k)}\right\}$ , whereas in reinforcement learning problems,  $\mathcal{D}=\left\{\left(\mathbf{x}_{1}, \hat{\mathbf{y}}_{1}, \ldots, \mathbf{x}_{\mathrm{H}}, \hat{\mathbf{y}}_{\mathrm{H}}\right)^{(\mathrm{k})}\right\}$ 

- $p(\mathbf{x_1})$ , a distribution over initial observations.

- $p\left(\mathbf{x}_{\mathrm{t}+1} \mid \mathbf{x}_{\mathrm{t}}, \mathbf{y}_{\mathrm{t}}\right)$ , a transition distribution.

- H, an episode length. In supervised learning problems, H $= 1$ . 



First, we need to consider a distribution over tasks $p(\mathcal{T})$ that we want our model to be able to adapt to, from where the training and testing tasks $\mathcal{T}_i$ are drawn. Generally, there are two phases, meta-training and meta-testing, where the tasks for meta-testing are held out during meta-training. During meta training, 



| Symbol | Definition  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| ------ | ------------------------------------------------------------|
|$\mathcal{T}$||








### Reference


[1] 

[2]

[3]


