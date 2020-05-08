---
title: "Thompson Sampling"
layout: post
author: Chunpai
tags: [reinforcement-learning]
typora-root-url: ./

---

Thompson sampling is a well-known multi-armed bandits algorithm for solving exploitation and exploration problem, which is also known as posterior sampling and probability matching. In this post, we will cover Bernoulli bandit problems, sequential recommendation, and reinforcement learning in MDPs with Thompson sampling. Most importantly, we will discuss when, why, and how to apply Thompson sampling. 


* TOC
{: toc}


## Bernoulli Bandit Problem with TS

Suppose there is a $K$-arms (actions) bandits, and each action yields either a success (reward $= 1$) or a failure (reward $= 0$). Action $$ k \in \{1, \cdots, K \} $$ has a hidden probability $\theta_k \in [0, 1]$ that produces a success, which is unknown to the agent, but fixed over time. The objective of the Bernoulli bandit is to learn or estimate the hidden probabilities for all arms by experimentation within $T$ periods, as well as to maximize the cumulative number of successes in $T$ periods.



### Beta-Bernoulli Bandit

Each $\theta_{k}$ can be interpreted as an action's success probability or mean reward. The mean rewards $\theta = (\theta_1, \cdots, \theta_K)$ are unknown, but fixed over time. For instance, an action $x_1$ is applied, and a reward $r_1 \in \{0,1\} $ is generated with success probability $P(r_1 = 1 \mid x_1, \theta) = \theta_{1}$ . 



Now, if we assume an independent beta prior distribution over each $\theta_k$ with hyper-parameters $\alpha = (\alpha_1, \cdots, \alpha_K)$ and $\beta = (\beta_1, \cdots, \beta_K)$, which could be mathematically formalized as:


$$
p\left(\theta_{k}\right)=\frac{\Gamma\left(\alpha_{k}+\beta_{k}\right)}{\Gamma\left(\alpha_{k}\right) \Gamma\left(\beta_{k}\right)} \theta_{k}^{\alpha_{k}-1}\left(1-\theta_{k}\right)^{\beta_{k}-1}
$$


It is very convenient to update each action's posterior distribution because of the beta-binomial conjugacy:


$$
\left(\alpha_{k}, \beta_{k}\right) \leftarrow\left\{\begin{array}{ll}
\left(\alpha_{k}, \beta_{k}\right) & \text { if } x_{t} \neq k \\
\left(\alpha_{k}, \beta_{k}\right)+\left(r_{t}, 1-r_{t}\right) & \text { if } x_{t}=k
\end{array}\right.
$$


The parameters $(\alpha_k, \beta_k)$ are sometimes called pseudo- counts, since $\alpha_k$ or $\beta_k$ increases by one with each observed success or failure, respectively. A beta distribution with parameters $(\alpha_k, \beta_k)$ has mean $\alpha_k/(\alpha_k, \beta_k)$, and the distribution becomes more concentrated as $(\alpha_k, \beta_k)$ grows. For example, the figure below



<img src="/../assets/img/beta_distribution.png" alt="Beta distribution over mean rewards." width="500" height="400"/>

 


### Algorithm 1
```latex
Just a sample algorithmn
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\KwResult{Write here the result}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Write here the input}
\Output{Write here the output}
\BlankLine
\While{While condition}{
    instructions\;
    \eIf{condition}{
        instructions1\;
        instructions2\;
    }{
        instructions3\;
    }
}
\caption{While loop with If/Else condition}
\end{algorithm} 
```


## General Thompson Sampling

## 

## Reference 

[1] 

[2] 




















