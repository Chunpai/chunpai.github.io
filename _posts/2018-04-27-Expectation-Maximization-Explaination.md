---
layout: post
tags: approximate-inference
author: Chunpai
---

I had written a note about the expectation maximization couple years ago. Recently, my research topic changes to the field of knowledge tracing, and many classic methods (i.e. BKT, SPARFA) leverage EM in their algorithm. I have to recap the EM algorithm and implement those baseline methods. In this post, I will go over some examples of using EM algorithm, such as *Gaussian Mixture Model (GMM)* and *Hidden Markov model (HMM)*. 

In order to understand the implementation throughly, it is better to read my note in the following link first: [Expectation Maximization](/assets/note/Expectation_Maximization_Explanation.pdf). Please leave comments if you find any issues or questions. Thanks.

* TOC
{: toc}
## Coin Tossing

Assume 3 coins A, B, C, and the probabilities of tossing head are unknown for these 3 coins, which are denoted as $\pi, p, q$ respectively. Now there is an experiment, we toss coin $A$ first. If $A$ is head, then toss $B$; if $A$ is tail, then toss $C$. We have 10 trails, and observed following results:

```
1,1,0,1,0,0,1,0,1,1
```

Now we would like to know if these results of coin $B$ or coin $C$, but we do not know the parameters $\pi, p, q$, we have to estimate them based on the observations. 

Let $X$ denotes the observed outcome, and $Y$ denotes the hidden variable, which is the tossing result of coin $A$. If there is only one observed data instead of 10, then we can write the likelihood as 


$$
\begin{align}
P(X ; \mathbb{\theta}) &= \sum_{Y=\{H,T\}} P(X, Y ; \mathbb{\theta}) \\
&= \sum_{Y=\{H,T\}} P(X\mid Y; \mathbb{\theta}) P(Y; \mathbb{\theta})\\
&= P(Y=H)P(X=x \mid Y=H) + P(Y=T)P(X=x\mid Y=T) \\
&= \pi \cdot p^x \cdot (1-p)^{1-x} + (1-\pi)\cdot q^{x} \cdot (1-q)^{1-x} 
\end{align}
$$


where $\mathbb{\theta}$ is $\{\pi, p, q\}$. If $A$ is head, the observation is from $B$, otherwise $C$. Now, we have observed data $X = \{x_1, \cdots, x_n\}$ then we have the likelihood as 




$$
P(X;\mathbb{\theta}) = \prod_{i=1}^{n}\pi \cdot p^{x_i} \cdot (1-p)^{1-x_{i}} + (1-\pi)\cdot q^{x_{i}} \cdot (1-q)^{1-x_{i}} 
$$


and our goal is to find the optimal $\hat{\mathbb{\theta}}$ 


$$
\begin{align}
 \hat{\theta} &= \arg\max_{\theta} L(\theta|X)\\
    &= \arg\max_{\theta} \log P(X|\theta) \\
    &= \arg\max_{\theta} \log \prod_{i=1}^{n} \left[\pi p^{x_i} (1-p)^{x_i} + (1-\pi) q^{x_i}(1-q)^{x_i} \right]\\
    &= \arg\max_{\theta} \sum_{i=1}^{n} \log \left[\pi p^{x_i} (1-p)^{x_i} + (1-\pi) q^{x_i}(1-q)^{x_i} \right]
\end{align}
$$


We cannot find the analytic form solution for $\hat{\mathbb{\theta}}$, and hence we have to leverage the EM algorithm. 

**E-Step:** assume at $(i-1)$ iteration, we have estimate $\mathbb{\theta}^{(i-1)} = \{ \pi^{(i-1)}, p^{(i-1)}, q^{(i-1)} \}$ , then we can write down the Q-function: 


$$
\begin{align}
Q(\theta,\theta^{(i-1)}) &= E_{Y|X} \big[\log P(X,Y;\theta) | X;\theta^{(i-1)}\big] \\
&= \sum_{j=1}^{n} \sum_{Y=\{H,T\}} P(Y|x_j;\theta^{(i-1)}) \log P(x_j,Y;\theta^{(i-1)}) \\
&= \sum_{j=1}^{n} \sum_{Y=\{H,T\}} P(Y|x_j;\theta^{(i-1)}) \log P(x_j,Y;\pi^{(i-1)}, p^{(i-1)},q^{(i-1)}) \\
&= \sum_{j=1}^{n} \sum_{Y=\{H,T\}} \frac{P(Y, x_j ; \theta^{(i-1)})}{P(x_j ; \theta^{(i-1)})} \log P(x_j,Y; \theta^{(i-1)})  \\
&= \sum_{j=1}^{n} \Bigg\{ \frac{\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j}}{\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j} + (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}}  \log [\pi p^{x_j} (1-p)^{x_j}] \\
&+ \frac{ (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}} {\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j} + (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}}  \log [(1-\pi) q^{x_j} (1-q)^{x_j}]  \Bigg\}
\end{align}
$$


**M-Step:** compute new parameter by taking derivative of the Q-function w.r.t different parameters. Here we denote:




$$
u_j = \frac{\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j}}{\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j} + (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}}
$$


and 


$$
1-u_j = \frac{ (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}} {\pi_{(i-1)} p_{(i-1)}^{x_j} (1-p_{(i-1)})^{1-x_j} + (1-\pi_{(i-1)}) q_{(i-1)}^{x_j} (1-q_{(i-1)})^{1-x_j}}
$$


Then, we take derivate of $Q$ w.r.t $\pi$ and set it to zero, that is 


$$
\frac{\partial Q}{\partial \pi} = \sum_{j=1}^{N} \big( \frac{u_j}{\pi} - \frac{1-u_j}{1-\pi}\big) = \sum_{j=1}^{N} \frac{u_j -\pi}{\pi(1-\pi)} = \frac{\sum_{j=1}^{N}u_j - N\cdot \pi}{\pi(1-\pi)} = 0  
$$


Hence, the new parameter $\pi$ at $(i)$ iteration is 


$$
\pi_{(i)} = \frac{1}{N}\sum_{j=1}^{N} u_j
$$


We can do the same thing to parameters $p$ and $q$, and we get 


$$
\begin{align}
    p_{(i)} &= \frac{\sum_{j=1}^{N} u_j x_j}{\sum_{j=1}^{N}u_j} \\
    q_{(i)} &= \frac{\sum_{j=1}^{N} (1-u_j) x_j}{\sum_{j=1}^{N} (1-u_j)}
\end{align}
$$


Thereafter, we could apply E-step and M-step iteratively until converged.












