---
layout: post
tags: generative-model
author: Chunpai
---

In this post, we will see the difference between generative models and discriminative models. And I will provide some examples of generative models, such as Naive Bayes (NB) and Gaussian Discriminative Analysis (GDA), and some discriminative models, such as Logistic Regression (LR) and Conditional Random Fields (CRF). 

* TOC
{: toc}
## Generative vs Discriminative Models

Generative models are models that describe how a label vector $\mathbf{y}$ can probabilistically "generate" a feature vector $\mathbf{x}$. Discriminative models work in the reverse direction, describing directly how to take a feature vector $\mathbf{x}$ and assign it a label $\mathbf{y}$. For example, some examples of generative models are 

- Naive Bayes
- Gaussian Discriminative Analysis
- Hidden Markov Models

and some examples of discriminative models are:

- Logistic regression
- Support vector machine
- Conditional random fields
- Traditional neural networks

The fundamental difference between these two models are:

- Discriminative model learn the (hard or soft) boundary between classes
- Generative models model the distribution of individual classes























