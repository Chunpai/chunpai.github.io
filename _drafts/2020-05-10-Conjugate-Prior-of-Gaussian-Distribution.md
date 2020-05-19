---
title: "Conjugate-Priors-of-Gaussian-Distribution"
layout: post
tags: [bayesian-inference]
author: Chunpai
---

[Updated on 05/10/2020.] 

* TOC
{: toc}


## Univariate Normal Distribution 



## Conjugate Prior of Univariate Normal Distribution



Likelihood of normal distribution could be written as 
$$
\begin{align}
f\left(x_{1},\cdots, x_{n} \mid \mu, \sigma^{2}\right) &= f\left(x_{n} \mid \mu, \sigma^{2}\right)=\prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi^{2}}} \exp \left\{-\frac{\left(x_{i}-\mu\right)^{2}}{2 \sigma^{2}}\right\} \\
&=(2 \pi)^{-\frac{n}{2}}\left(\sigma^{2}\right)^{-\frac{n}{2}} \exp\left\{-\frac{1}{\sigma^2} \cdot \sum_{i=1}^{n} \frac{(x_{i}-\mu)^{2}}{2})\right\}
\end{align}
$$


### Unknown Mean, but Known Variance: *Normal Distribution*



If the data $x_1, \cdots, x_n$ are independent and identically distributed from a normal distribution where the mean $\mu_X$ is unknown and the variance $\sigma_X^2$ is known. The likelihood could be rewritten as proportion to 


$$
L\left(\mu_X | x_{1}, \ldots, x_{n}\right) \propto \exp \left(-\frac{n}{2\sigma_X^2}(\bar{x}-\mu_X)^{2}\right)
$$


where the sufficient statistics are $n$, the number of data points, and $\bar{x}$ , the mean of the data. The conjugate prior is a normal distribution with mean $\nu_{\mu}$ and variance $\sigma_{\mu}^2$ :


$$
\pi(\mu \mid \nu_{\mu}, \sigma_{\mu}^2) = \sqrt{\frac{1}{2\pi \sigma_{\mu}^2}} \exp\left( -\frac{1}{2\sigma_{\mu}^2}(\mu - \nu_{\mu})^2\right) \propto \exp\left( -\frac{1}{2\sigma_{\mu}^2}(\mu - \nu_{\mu})^2\right)
$$


Therefore,  the posterior is proportion to 


$$
\begin{align}
p(\mu \mid \cdot) &\propto \exp \left(-\frac{n}{2\sigma_X^2}(\bar{x}-\mu_X)^{2}\right) \cdot \exp\left( -\frac{1}{2\sigma_{\mu}^2}(\mu - \nu_{\mu})^2 \right) \\
&= \exp \left(-\frac{n\cdot \sigma_{\mu}^2}{2\sigma_X^2 \cdot \sigma_{\mu}^2}(\bar{x}-\mu_X)^{2}\right) \cdot \exp\left( -\frac{\sigma_X^2}{2\sigma_X^2\cdot \sigma_{\mu}^2}(\mu - \nu_{\mu})^2 \right) \\
&= \exp \left(-\frac{n\cdot \sigma_{\mu}^2}{2\sigma_X^2 \cdot \sigma_{\mu}^2}(\bar{x}-\mu_X)^{2} -\frac{\sigma_X^2}{2\sigma_X^2\cdot \sigma_{\mu}^2}(\mu - \nu_{\mu})^2 \right) \\
&= \exp \left(- \left[ \frac{n\cdot \sigma_{\mu}^2(\bar{x}-\mu_X)^{2} + \sigma_X^2 (\mu - \nu_{\mu})^2}{2\sigma_X^2\cdot \sigma_{\mu}^2} \right] \right) \\
&= \exp \left(- \left[ \frac{\sigma_{\mu}^2 (n\bar{x}^2-2\bar{x}\mu_X + \mu_X^2) + \sigma_X^2 (\mu^2 - 2\mu\nu_{\mu} + \nu_{\mu}^2)}{2\sigma_X^2\cdot \sigma_{\mu}^2} \right] \right)\\
&= 
\end{align}
$$


With some algebra, we could derive the posterior distribution is normal distributed with mean $\tilde{\mu}$ and variance $\tilde{\sigma}^2$ as


$$
\tilde{\mu} = \tilde{\sigma}^2 \left(\frac{\nu_{\mu}}{\sigma_{\mu}^2} + \frac{n\bar{x}}{\sigma_X^2} \right)
$$

$$
\tilde{\sigma}^2 = \frac{\sigma_{\mu}^2\sigma_X^2}{\sigma_{\mu}^2 + \sigma_{X}^2}
$$




### Known Mean, but Unknown Variance: *Inverse Gamma Distribution*

The inverse gamma distribution is 



$$
g(y ; \alpha, \beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)} y^{-(\alpha+1)} e^{-\beta / y}
$$



Since the variance of Normal distribution is unknown, and say that the prior of variance follows the inverse gamma distribution, that is we let $y=\sigma^2$ . 



$$
g(\sigma^2; \alpha, \beta) =\frac{\beta^{\alpha}}{\Gamma(\alpha)} (\sigma^2)^{-(\alpha+1)} e^{-\beta / \sigma^2}
$$



Since the posterior is proportion to the likelihood times the prior, thus we could **ignore all constants (w.r.t $\sigma^2$)** in both likelihood and prior. Therefore, we have the form of posterior distribution as



$$
\begin{align}
posterior &\propto
(\sigma^2)^{-(\alpha+1)} \cdot e^{-\beta / \sigma^2} \cdot \left(\sigma^{2}\right)^{-n/2} \exp\left\{-\frac{1}{\sigma^2} \cdot \sum_{i=1}^{n} \frac{(x_{i}-\mu)^{2}}{2})\right\} \\
&= (\sigma^2)^{-(\alpha+\frac{n}{2}+1)} \cdot \exp\left\{-\frac{1}{\sigma^2} \cdot \left( \beta + \sum_{i=1}^{n} \frac{(x_{i}-\mu)^{2}}{2})\right) \right\} 
\end{align}
$$



We can see that this form looks very similar to the inverse gamma distribution without the normalization term. Since the integral of posterior distribution on $ \sigma^2$ should be equal to 1, and the inverse gamma distribution with constant term $Z$ is also equal to 1:



$$
\int_{\sigma^2=0}^{\infty} Z \cdot (\sigma^2)^{-(\alpha+\frac{n}{2}+1)} \cdot \exp\left\{-\frac{1}{\sigma^2} \cdot \left( \beta + \sum_{i=1}^{n} \frac{(x_{i}-\mu)^{2}} {2})\right)\right\}  d\sigma^2 = 1.0
$$



Therefore, we can say that the posterior distribution is the inverse gamma distribution with parameters $(\alpha + \frac{n}{2}, \beta + \sum_{i=1}^{n} \frac{(x_{i}-\mu)^{2}} {2})$, where $\mu$ is the known mean of normal distribution.



### Unknown Mean and Unknown Variance: 



## Multivariate Normal Distribution 

## Conjugate Prior of Multivariate Normal Distribution




## Log-normal Distribution 

The **log-normal distribution** is the probability distribution of a random variable whose logarithm follows a normal distribution. It models phenomena whose **relative growth rate** is independent of size, which is true of most natural phenomena including the size of tissue and blood pressure, income distribution, and even the length of chess games.

Let *Z* be a **standard normal variable**, which means the probability distribution of *Z* is normal centered at 0 and with variance 1. Then a log-normal distribution is defined as the probability distribution of a random variable

$$
X = e^{\mu+\sigma Z},
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the **logarithm of** *X*, respectively. The term "log-normal" comes from the result of taking the logarithm of both sides: 


$$
\log X = \mu +\sigma Z
$$



Therefore, the definition of log-normal distribution is 
$$
f_X(x) = \frac{1}{x}\frac{1}{\sigma \sqrt{2\pi}}e^{-\dfrac{(\ln x-\mu)^2}{2\sigma^2}},
$$

The mean and variance of $X$ could be derived as 

$$
\mathbb{E}[X] = \exp(\mu + \frac{\sigma^2}{2})
$$

$$
Var[X] = (\exp(\sigma^2) - 1) \exp(2\mu + \sigma^2) = (\exp(\sigma^2) - 1) \cdot  \mathbb{E}^2[X]
$$

## Conjugate Prior of Log-Normal Distribution 


### 

## Reference 

[1] Fink, Daniel. "A Compendium of Conjugate Priors." (1997).

[2] https://jrnold.github.io/bayesian_notes/priors.html#levels-of-priors






















