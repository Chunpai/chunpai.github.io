---
layout: post
tags: approximate-inference
author: Chunpai
---

This post is just a quick explanation of linear regression and Bayesian linear regression from the probabilistic perspective.

* TOC
{: toc}
## Linear Regression

Given data $\mathcal{D} = \{\mathbb{X},\mathbb{y}\}$ where $\mathbb{X}\in \mathbb{R}^{m\times n}$ and $\mathbb{y} \in \mathbb{R}^{m\times 1}$ ,  we assume the hypothesis is in the form 

  
$$
h_{\theta}(\mathbb{x}) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \sum_{i=0}^{n} \theta_i x_i = \mathbb{\theta}^\top\mathbb{x}\ \ \text{  where  } x_0 = 1
$$


then the objective function of linear regression is 


$$
\arg\min_{\mathbb{\theta}} J(\mathbb{\theta}) = \frac{1}{2}\sum_{i=1}^{m} \left( h_{\mathbb{\theta}}(\mathbb{x}^{(i)}) - y^{(i)} \right)
$$


### Probabilistic Interpretation

Assume target variables and the inputs are related via the equation:


$$
y^{(i)} = \mathbb{\theta}^\top \mathbb{x}^{(i)} + \epsilon^{(i)}
$$


where $\epsilon^{(i)}$ is an error term that capturers either unmodeled effects (some important pertinent features we left out) or random noise. Further, we assume $\epsilon^{(i)}$ are distributed I.I.D and 


$$
\epsilon^{(i)}\sim \mathcal{N}(0, \sigma^2)
$$


Thus, we could write the probability of the error term as


$$
P(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2} \right)
$$


Since $\epsilon^{(i)}$ is a random variable, thus $y^{(i)}$ is also a random variable, which implies that


$$
P(y^{i} \mid \mathbb{x}^{(i)}; \mathbb{\theta}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right)
$$


That is 


$$
y^{i} \mid \mathbb{x}^{(i)}; \mathcal{\theta} \sim \mathcal{N} \left(\mathbb{\theta}^\top \mathbb{x}^{(i)}, \sigma^2 \right)
$$


Therefore, we could write the likelihood function of all data $\mathcal{D} = \{(\mathbb{x}^{(i)}, y^{(i)})\}_m$ as 

