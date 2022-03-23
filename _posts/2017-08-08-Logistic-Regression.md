---
layout: post
tags: structured-prediction
author: Chunpai
---

This post is about some key points of logistic regression. Logistic regression is a commonly used classification method with advantages such as easy implementation and interpretability. 

* TOC
{: toc}


# Logistic Function

**The standard logistic function:** 

$$
\sigma(x) = \frac{1}{1+e^{-x}}  
$$
which maps $x\in (-\infty, \infty)$ to $(0,1)$ that could be interpreted as probability or cumulative probability.


We could easily find that $\sigma(-x) = 1-\sigma(x)$. 

$$
\displaystyle 
\begin{aligned}
\sigma(-x) &=  \frac{1}{1+e^{x}} = \frac{1}{1+\frac{1}{e^{-x}}} 
= \frac{e^{-x}}{1+e^{-x}} \\
&= 1 - \sigma(x)
\end{aligned}
$$

Since $\sigma(x) \in (0,1)$, we could view the output of sigmoid function as a probability of an event happening
$$
p = \frac{1}{1+e^{-x}}
$$
and the probability of an event not happening is
$$
1 -p = \frac{e^{-x}}{1+e^{-x}}  
$$

The odds ratio is defined as 
$$
\frac{p}{1-p} = \frac{1}{e^{-x}} = e^{x}
$$
and we could have the log-odds as 
$$
\ln(\frac{p}{1-p}) = \ln(e^{x}) = x
$$

We could see that **the input of standard logistic function is actually the log-odds**.

The **inverse of logistic function** could be used to map the value from $(0,1)$ to real values. If we denote the value $p=\sigma(x)$, then
$$
\sigma^{-1}(x) = \ln(\frac{p}{1-p})
$$
which is just the log-odds and also called **logit function**.

## Logit and Probit
Since the logit function has the domain between 0 and 1, it commonly used as the inverses of the cumulative distribution function (CDF) of the logistic distribution.

The probit function is closely related to the logit function. Both logit and probit are sigmoid function (S-shape curve). The probit is the inverse CDF of the normal distribution, which is denoted as $\Phi^{-1}(x)$, where 

$$
\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-(z^2/2)}dz
$$

The probit function is sometimes used in place of logit function because for certain applications (e.g. Bayesian statistics) the implementation is easier, because in Bayesian, the conjugate-prior of normal distribution is normal distribution. 


## Derivative of Logistic Function

$$
\begin{aligned}
\sigma^{'}(x) &= \frac{d}{dx}\frac{1}{1+e^{-x}} \\
&= \frac{1}{(1+e^{-x})^2}\cdot e^{-x}\\
&= \frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}}\\
&= \sigma(x) (1-\sigma(x))
\end{aligned}
$$


# Binomial Logistic Regression
The binomial loigsitic regression is a discriminative classification model with conditional probability, 
$p(y\mid \mathbf{x}, \bm{\theta})$, that is 

$$
p(y\mid \mathbf{x}, \bm{\theta}) = Ber(y\mid \sigma(\mathbf{w}^\top\mathbf{x} + b))
$$


where 
* $\sigma$ denotes the sigmiod function,
*  $\mathbf{x}\in\mathbb{R}^D$ is the D-dimensional input vector,
*  $y\in\{-1, 1\}$ is the class label,
*  $\bm{\theta}$ denotes all parameters including weights $\mathbf{w}$ and bias $b$.

The probability of each label could be computed as 
$$
p(y\mid \mathbf{x}, \bm{\theta}) = \sigma(y\cdot (\mathbf{w}^\top\mathbf{x}+b))
$$
with 
$$\sigma(-(\mathbf{w}^\top\mathbf{x}+b)) = 1 - \sigma(\mathbf{w}^\top\mathbf{x}+b)
$$

When we use label $y\in \{0,1\}$, the probability of each label is 
$$
p(y\mid \mathbf{x}, \bm{\theta}) = (\sigma(\mathbf{w}^\top\mathbf{x}+b))^{y}(1-\sigma(\mathbf{w}^\top\mathbf{x}+b))^{(1-y)}
$$



Based on the previous section, we found that the input of standard logistic function is the log-odds. Therefore, the quantity $a=\mathbf{w}^\top\mathbf{x}+b$ is the log-odds (also called the logit or the pre-activation in ML). 

Notice that, despite logistic function provides non-linear mapping from $(-\infty, \infty)$ to $(0,1)$, the logistic regression is still considered as a generalized linear model, because the predicted probability is always depends on the sum of the inputs and parameters (e.g. $w_1x_1 + w_2x_2 +\cdots$ ). 
In other words, the predicted probability does not depend on interaction between the features (e.g. $w_1x_1x_2 + w_2x_2x_3$).


















