---
layout: post
tags: reinforcement-learning
author: Chunpai
---

This post is about two methods related to deep Q learning in continuous action space, which are DDPG and NAF. 


* TOC
{: toc}


## Policy Objective Functions

In previous post of stochastic policy gradient, we have formulate the goal of reinforcement learning as finding the optimal policy $\pi$ which is parameterized by $$\theta$$ such that have *maximum expected return*:


$$
\theta^{*} = \arg \max _{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T} r\left(s_{t}, a_{t}\right)\right]
$$


The objective above can also be viewed as ***maximum expected start value*** in episodic environments:


$$
J_{0}(\theta)=\mathbb{E}_{\pi_{\theta}}\left[G_{0} \right] = \sum_{s_0 \in \mathcal{S}} p_{1}(s_0) \cdot V^{\pi_{\theta}}(s_0)
$$


where $p_{1}$ denotes the starting probability and $G_0 = R_1 + \cdots + R_{T}$, which is independent of the policy $ \pi$. However, in continuous tasks, we use the ***average state value*** as objective function:


$$
J_{avV}(\theta)=\sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) V^{\pi_{\theta}}(s) = \sum_{s \in \mathcal{S}} d^{\pi_{\theta}}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a \mid s) Q^{\pi_{\theta}}(s, a)
$$


or the ***average reward per time-step***:


$$
J_{a v R}(\theta)=\sum_{s} d^{\pi_{\theta}}(s) \sum_{a} \pi_{\theta}(s, a) r(s, a)
$$


where $d^{\pi_{\theta}}(s)$ is *stationary distribution of Markov chain* for $\pi_{\theta}$. We also have 


$$
\begin{align}
d^{\pi_{\theta}}(s) &= \sum_{s^{\prime} \in \mathcal{S}} d^{\pi_{\theta}} \left(s^{\prime}\right) \mathcal{P}_{s^{\prime} s}\\
&= \sum_{s^{\prime} \in \mathcal{S}}\sum_{t=1}^{\infty} p_{1}(s') p(s' \rightarrow s, t, \pi_{\theta}) 
\end{align}
$$
where $p(s' \rightarrow s, t, \pi_{\theta}) $ denotes the probability from state $s'$ to state $s$ with $t$ steps transition. In addition, we sometimes introduce the discount factor $\gamma \in (0, 1)$ into MDP, that is every state transition probability times $\gamma$, 


$$
\tilde{p}(s'\rightarrow s, \color{red}{t=1}, \pi_{\theta}) = \gamma \cdot p(s' \rightarrow s, \color{red}{t=1}, \pi_{\theta})
$$


and since $\sum_{s'} p(s' \rightarrow s, t, \pi_{\theta}) = 1.0$, the discount factor in fact changes the MDP and add one extra terminate state *implicitly* with probability $ 1-\gamma$, which guarantees that $\sum_{s'} \tilde{p}(s' \rightarrow s, t, \pi_{\theta}) = 1.0$ ; Thus, we also have discounted state distribution as: 


$$
d^{\pi_{\theta}}(s) = \sum_{s^{\prime} \in \mathcal{S}}\sum_{t=1}^{\infty} \gamma^{t-1}\cdot p_{1}(s') p(s' \rightarrow s, t, \pi_{\theta})
$$


## Policy Gradient Theorem

The [policy gradient theorem](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) [4] states that the policy gradient is not dependent on *the gradient of state distribution* $\nabla_{\theta} d^{\pi_{\theta}}(s)$, despite the state distribution depends on the policy parameter $\theta$. This theorem makes computing policy gradient possible. We will show the proof in terms of the state-value function at below.

> $$
> \nabla J(\theta) \propto \sum_{s\in \mathcal{S}} d(s) \sum_{a} \nabla \pi(a \mid s) Q_{\pi}(s, a)
> $$
>
> 
>
> Proof. First we show the gradient of state-value function
>
> 
> $$
> \begin{aligned} \nabla V_{\pi}(s) 
> &=\nabla\left[\sum_{a} \pi(a | s) Q_{\pi}(s, a)\right], \quad \text { for all } s \in \delta \\ &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \nabla Q_{\pi}(s, a)\right] \\ &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \nabla \sum_{s^{\prime} r} p\left(s^{\prime}, r | s, a\right)\left(r+V_{\pi}\left(s^{\prime}\right)\right)\right] \\
> &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \sum_{s^{\prime}} p\left(s^{\prime} | s, a\right) \nabla V_{\pi}\left(s^{\prime}\right)\right] \\ 
> &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \sum_{s^{\prime}} p\left(s^{\prime} | s, a\right) \sum_{a^{\prime}} [\nabla \pi\left(a^{\prime} | s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)+\pi\left(a^{\prime} | s^{\prime}\right) \sum_{s^{\prime \prime}} p\left(s^{\prime \prime} | s^{\prime}, a^{\prime}\right) \nabla v_{\pi}\left(s^{\prime \prime}\right) ] \right] \\
> &= \sum_{x \in S} \sum_{k=0}^{\infty} \operatorname{Pr}(s \rightarrow x, k, \pi) \sum_{a} \nabla \pi(a | x) Q_{\pi}(x, a) 
> \end{aligned}
> $$
>  

## Deterministic Policy Gradient (DPG)

You can see that the $$\nabla_{\theta} J(\theta)$$ derived in stochastic policy gradient becomes 0 if it is deterministic policy. 



## Deep Deterministic Policy Gradient (DDPG)



## Normalized Advantage Function (NAF)





## Reference

[1] [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf )

[2] [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf )

[3] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748v1.pdf )

[4] [Policy Gradient Methods for Reinforcement Learning with FunctionApproximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) 




