---
title: "Probabilistic Interpretation of Ridge Regression and LASSO"
layout: post
tags: [probabilistic-machine-learning, generalized-linear-model, statistics]
author: Chunpai
---

This post is just a quick explanation of linear regression, ridge regression, and LASSO from the probabilistic perspective.

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

$$
\begin{aligned}
P(\mathbb{y}\mid \mathbb{X}; \mathbb{\theta}) &= L(\mathbb{\theta}; \mathbb{X}, \mathbb{y}) \\
&= \prod_{i=1}^{m} P(y^{i} \mid \mathbb{x}^{(i)}; \mathbb{\theta}) \\
&= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right)
\end{aligned}
$$

We could maximize the log-likelihood to find best $\mathbb{\theta}$ that fit the data with high probability, hence we have

$$
\begin{aligned}
\arg\max_{\mathbb{\theta}} L(\mathbb{\theta}; \mathbb{X}, \mathbb{y}) &= \arg\max_{\mathbb{\theta}} \log L(\mathbb{\theta}; \mathbb{X}, \mathbb{y}) \\
&=\arg\max_{\mathbb{\theta}} \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right) \\
&=\arg\max_{\mathbb{\theta}} \sum_{i=1}^{m} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right) \right] \\
&=\arg\max_{\mathbb{\theta}} \sum_{i=1}^{m} \left[ \log \frac{1}{\sqrt{2\pi\sigma^2}}  + \log \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right)\right]\\
&=\arg\max_{\mathbb{\theta}} \sum_{i=1}^{m} \left[ \log \frac{1}{\sqrt{2\pi\sigma^2}}  - \frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right] \\
&=\arg\max_{\mathbb{\theta}} \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi\sigma^2}} - \sum_{i=1}^{m}\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \\
&= \arg\max_{\mathbb{\theta}} -\sum_{i=1}^{m}\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \\
&= \arg\min_{\mathbb{\theta}} \frac{1}{2} \sum_{i=1}^{m}(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2
\end{aligned}
$$

* `<span style="color:blue">` We can see that minimizing the least square loss is same as maximizing the likelihood in linear regression under the assumption that $\epsilon^{(i)}\sim \mathcal{N}(0, \sigma^2)$.
* Notice that, $\sigma^2$ is also the parameter we need to find. Thus, we also need to use the MLE w.r.t $\sigma^2$.

## Ridge Regression and LASSO

The objective of ridge regression is

$$
\arg\min_{ \mathbb{\theta} } \frac{1}{2} \| \mathbb{y} - \mathbb{\theta}^\top\mathbb{X} \|^2_2 + \lambda \| \mathbb{\theta} \|_2^2
$$

and the objective of the least absolute shrinkage and selection operator (LASSO)  is

$$
\arg\min_{\mathbb{\theta}} \frac{1}{2} \|\mathbb{y} - \mathbb{\theta}^\top\mathbb{X}\|^2_2 + \lambda \|\mathbb{\theta}\|_1
$$

We will show that ridge regression which imposes $\ell_2$ regularization on least square cost function is just maximum a posterior (MAP) with Gaussian prior; LASSO which impose $\ell_1$ regularization on least square cost function is just MAP with Laplacian prior.

### Maximum A Posterior

$$
\begin{aligned}
\mathbb{\theta}_{\text{MAP}} &= \arg\max_{\mathbb{\theta}} P(\mathbb{\theta} \mid \mathcal{D}) \\
&= \arg\max_{\mathbb{\theta}} \frac{P(\mathbb{\theta})P(\mathcal{D}\mid \mathbb{\theta})}{P(\mathcal{D})}  \\
&= \arg\max_{\mathbb{\theta}} P(\mathbb{\theta})P(\mathcal{D}\mid \mathbb{\theta})\\
&= \arg\max_{\mathbb{\theta}} \log\left[ P(\mathbb{\theta})P(\mathcal{D}\mid \mathbb{\theta})\right] \\
&= \arg\max_{\mathbb{\theta}}  \log P(\mathcal{D}\mid \mathbb{\theta}) + \log P(\mathbb{\theta}) 
\end{aligned}
$$

### Probabilistic Interpretation of Ridge Regression

Assume the prior follows the Gaussian distribution, that is

$$
\mathbb{\theta} \sim \mathcal{N}(0, \tau^2)
$$

Then, we could write the down the MAP as

$$
\begin{aligned}
\mathbb{\theta}_{\text{MAP}} &= \arg\max_{\mathbb{\theta}}  \log P(\mathcal{D}\mid \mathbb{\theta}) + \log P(\mathbb{\theta})\\
&= \arg\max_{\mathbb{\theta}} \log \left[ \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right)  \right] + \log \left[\prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\tau^2}} \exp(-\frac{\mathbb{\theta}^2}{2\tau^2}) \right] \\
&= \arg\max_{\mathbb{\theta}} \left[-\sum_{i=1}^{m} \frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} - \sum_{i=1}^{m}\frac{\theta_j^2}{2\tau^2} \right] \\
&= \arg\max_{\mathbb{\theta}} -\frac{1}{\sigma^2} \left[\sum_{i=1}^{m} \frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2} + \frac{\sigma^2}{2\tau^2}\sum_{i=1}^{m}\theta_j^2 \right] \\
&= \arg\min_{\mathbb{\theta}} \frac{1}{2}\sum_{i=1}^{m} (y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2 + \frac{\sigma^2}{2\tau^2}\sum_{i=1}^{m}\theta_j^2 \\
&= \arg\min_{\mathbb{\theta}} \|\mathbb{y} - \mathbb{\theta}^\top \mathbb{X}\|_2^2 + \lambda \|\mathbb{\theta}\|_2^2
\end{aligned}
$$

where $\lambda = \frac{\sigma^2}{2\tau^2}$.

### Probabilistic Interpretation of LASSO

**Laplace Distribution:** if random variable $Z \sim Laplace(\mu, b)$, then we have

$$
P(Z = z \mid \mu, b) = \frac{1}{2b} \exp \left( -\frac{|z-\mu|}{b}\right)
$$

Now, let's assume the prior follows the Laplace distribution as described above, that is

$$
\mathbb{\theta} \sim Laplace(\mathbb{\mu}, b) \ \ \ \ \ \ \ \ P(\mathbb{\theta}) = \frac{1}{2b} \exp\left(-\frac{|\mathbb{\theta} - \mathbb{\mu}|}{b} \right)
$$

$$
\begin{aligned}
\mathbb{\theta}_{\text{MAP}} &= \arg\max_{\mathbb{\theta}}  \log P(\mathcal{D}\mid \mathbb{\theta}) + \log P(\mathbb{\theta})\\
&= \arg\max_{\mathbb{\theta}} \log \left[ \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} \right)  \right] + \log \left[\prod_{i=1}^{m} \frac{1}{2b} \exp(-\frac{|\mathbb{\theta}|}{b}) \right] \\
&= \arg\max_{\mathbb{\theta}} \left[-\sum_{i=1}^{m} \frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2\sigma^2} - \sum_{i=1}^{m}\frac{|\theta_j|}{b} \right] \\
&= \arg\max_{\mathbb{\theta}} -\frac{1}{\sigma^2} \left[\sum_{i=1}^{m} \frac{(y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2}{2} + \frac{\sigma^2}{b}\sum_{i=1}^{m}|\theta_j| \right] \\
&= \arg\min_{\mathbb{\theta}} \frac{1}{2}\sum_{i=1}^{m} (y^{(i)} - \mathbb{\theta}^\top \mathbb{x}^{(i)})^2 + \frac{\sigma^2}{b}\sum_{i=1}^{m}|\theta_j| \\
&= \arg\min_{\mathbb{\theta}} \|\mathbb{y} - \mathbb{\theta}^\top \mathbb{X}\|_2^2 + \lambda \|\mathbb{\theta}\|_1
\end{aligned}
$$

where $\lambda = \frac{\sigma^2}{b}$.

## Bayesian Linear Regression

From Bayesian view, there is uncertainty in our choice of parameters $\mathbb{\theta}$, and there should have a probability in $\mathbb{\theta}$, typically

$$
\mathbb{\theta} \sim \mathcal{N}(0, \tau^2 I)
$$

Based on the Baye's rule, we can write down the parameter posterior as

$$
\begin{aligned}
P(\mathbb{\theta}\mid \mathcal{D}) &= \frac{P(\mathbb{\theta})P(\mathcal{D}\mid \mathbb{\theta})}{P(\mathcal{D})}  \\
&= \frac{P(\mathbb{\theta})P(\mathcal{D}\mid \mathbb{\theta})}{\int_{\mathbb{\hat{\theta}}} P(\mathbb{\hat{\theta}}) P(\mathcal{D}\mid \mathbb{\hat{\theta}})d\mathbb{\hat{\theta}}}
\end{aligned}
$$

For Bayesian linear regression, the input is a testing data $ \mathbb{x}^* $, but the output is a probability distribution over $y^*$ instead of a numerical value:

$$
P(y^* \mid \mathbb{x}^*, \mathcal{D}) = \int_{\mathbb{\theta}} P(y^* |\mathbb{x}^*, \mathbb{\theta})\cdot P(\mathbb{\theta}\mid \mathcal{D}) d\mathbb{\theta}
$$
