---
layout: post
tags: reinforcement-learning
author: Chunpai
---

This post is about policy gradient theorem, generalized advantaged estimation, and two methods related to deep Q learning in continuous action space, which are DDPG and NAF. 


* TOC
{: toc}


## Policy Objective Functions

In previous post of stochastic policy gradient, we have formulate the goal of reinforcement learning as finding the optimal policy $\pi$ which is parameterized by $$\theta$$ such that have *maximum expected return*:


$$
\theta^{*} = \arg \max _{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T} r\left(s_{t}, a_{t}\right)\right]
$$


The objective above can also be viewed as ***maximum expected start value*** in episodic environments:


$$
J_{0}(\theta)=\mathbb{E}_{\pi_{\theta}}\left[G_{0} \mid \pi  \right] = \sum_{s_0 \in \mathcal{S}}p_{1}(s_0) \cdot V^{\pi_{\theta}}(s_0)
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

The [policy gradient theorem](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) [4] states that **the policy gradient is not dependent on the gradient of state distribution $\nabla_{\theta} d^{\pi_{\theta}}(s)$, despite the state distribution depends on the policy parameter $\theta$ **. This theorem makes computing policy gradient possible. We will show the proof in terms of the state-value function at below.

> $$
> \nabla J(\theta) \propto \sum_{s\in \mathcal{S}} d(s) \sum_{a} \nabla_{\theta} \pi_{\theta}(a \mid s) Q_{\pi}(s, a)
> $$
>
> 
>
> Proof. First we show the gradient of state-value function
>
>
> $$
> \begin{aligned} \nabla V_{\pi}(s) 
> &=\nabla\left[\sum_{a} \pi(a | s) Q_{\pi}(s, a)\right], \quad \text { for all } s \in \mathcal{S} \\ &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \nabla Q_{\pi}(s, a)\right] \\ &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \nabla \sum_{s^{\prime} r} p\left(s^{\prime}, r | s, a\right)\left(r+V_{\pi}\left(s^{\prime}\right)\right)\right] \\
> &=\sum_{a}\left[\nabla \pi(a | s) Q_{\pi}(s, a)+\pi(a | s) \sum_{s^{\prime}} p\left(s^{\prime} | s, a\right) \color{blue}{\nabla V_{\pi}\left(s^{\prime}\right)} \right] \\ 
> &= \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{a} \left[ \pi(a | s) \sum_{s^{\prime}} p\left(s^{\prime} | s, a\right) \color{blue}{\nabla V_{\pi}\left(s^{\prime}\right)} \right] \\
> &= \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}} \color{green}{\sum_{a}  \pi(a | s) p\left(s^{\prime} | s, a\right)} \color{blue}{\nabla V_{\pi}\left(s^{\prime}\right)}  \\
> &= \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}} \color{green}{p(s' \mid s)} \color{blue}{\nabla V_{\pi}\left(s^{\prime}\right)}  \\
> &= \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}} \color{green}{p(s\rightarrow s', t=1,\pi)} \color{blue}{\nabla V_{\pi}\left(s^{\prime}\right)}  \\
> &=\sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}} \color{green}{p(s\rightarrow s', t=1,\pi)} \color{blue}{\sum_{a^{\prime}} [\nabla \pi\left(a^{\prime} | s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)+\pi\left(a^{\prime} | s^{\prime}\right) \sum_{s^{\prime \prime}} p\left(s^{\prime \prime} | s^{\prime}, a^{\prime}\right) \nabla V_{\pi}\left(s^{\prime \prime}\right)]}  \\
> &=\sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}} \color{green}{ p(s\rightarrow s', t=1,\pi)} \color{blue}{ \sum_{a^{\prime}} \nabla \pi\left(a^{\prime} | s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)}+ \sum_{s^{\prime}} \color{green}{ p(s\rightarrow s', t=1,\pi)} \color{blue}{ \sum_{a^{\prime}} \pi\left(a^{\prime} | s^{\prime}\right) \sum_{s^{\prime \prime}} p\left(s^{\prime \prime} | s^{\prime}, a^{\prime}\right) \nabla V_{\pi}\left(s^{\prime \prime}\right)}  \\
> &= \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}}  p\left(s \rightarrow s^{\prime}, t=1, \pi\right) \sum_{a^{\prime}}\nabla \pi\left(a^{\prime} | s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)+ \color{blue}{\sum_{s^{\prime\prime}}  p\left(s \rightarrow s^{\prime\prime }, t=2, \pi\right) \nabla V_{\pi}\left(s^{\prime \prime}\right)} \\
> &= \underbrace{\sum_{s}  p(s\rightarrow s, t = 0, \pi)}_{=1.0} \sum_{a}\nabla \pi(a | s) Q_{\pi}(s, a)+ \sum_{s^{\prime}}  p\left(s \rightarrow s^{\prime}, t=1, \pi\right) \sum_{a^{\prime}}\nabla \pi\left(a^{\prime} | s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)+ \color{blue}{\sum_{s^{\prime\prime}}  p\left(s \rightarrow s^{\prime\prime }, t=2, \pi\right) \nabla V_{\pi}\left(s^{\prime \prime}\right)} \\
> &= \text{recursively unrolling the formula . . .} \\
> &= \sum_{x \in S} \sum_{t=0}^{\infty} \operatorname{Pr}(s \rightarrow x, t, \pi) \sum_{a} \nabla \pi(a | x) Q_{\pi}(x, a) 
> \end{aligned}
> $$
>
> Now, we can compute the gradient of objective function:
>
> 
>
> $$
> \begin{aligned} 
> \nabla_{\theta} J(\theta) 
> &= \sum_{s_0 \in \mathcal{S}} p_1(s_0) \nabla_{\theta} V^{\pi}\left(s_{0}\right) \\ 
> &= \sum_{s_0 \in \mathcal{S}} p_1(s_0) \sum_{s \in \mathcal{S}} \sum_{t=0}^{\infty} p\left(s_{0} \rightarrow s, t, \pi\right) \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)  \\ 
> &= \sum_{s \in \mathcal{S}} \left[ \sum_{s_0 \in \mathcal{S}}  \sum_{t=0}^{\infty} p_1(s_0) p\left(s_{0} \rightarrow s, t, \pi\right)\right] \sum_{a} \nabla \pi(a | s) Q_{\pi}(s, a)  \\
> &=  \sum_{s \in \mathcal{S}} d^{\pi}(s) \sum_{a} \nabla_{\theta} \pi_{\theta}(a | s) Q^{\pi}(s, a) \\
> &= \sum_{s \in \mathcal{S}} d^{\pi}(s) \sum_{a} \pi_{\theta}(a | s)  \frac{\nabla_{\theta} \pi_{\theta}(a | s)}{\pi_{\theta}(a | s) } Q^{\pi}(s, a) \\
> &= \sum_{s \in \mathcal{S}} d^{\pi}(s) \sum_{a} \pi_{\theta}(a | s)  \nabla_{\theta} \log \pi_{\theta}(a | s)Q^{\pi}(s, a) \\
> &= \sum_{s \in \mathcal{S}} \sum_{a} \left[ d^{\pi}(s)  \pi_{\theta}(a | s) \right] \nabla_{\theta} \log \pi_{\theta}(a | s)Q^{\pi}(s, a) \\
> &= \operatorname{E}_{s \sim d^{\pi}, a \sim A(s)} \left[ \nabla_{\theta} \log \pi_{\theta}(a | s)Q^{\pi}(s, a) \right]
> \end{aligned}
> $$
>
> 
>
> The last equation above can be interpreted as the expected value of $  \nabla_{\theta} \log \pi_{\theta}(a \mid s)Q^{\pi}(s, a) $ for a state and an action pair with stationary distribution. However, the equation above depends on the state stationary distribution, which depends on the transition probability, and in most case we do not know the transition probability, and we have to sample a lot trajectories to estimate it ? How can we change the last equation to the generalized advantage estimation [6] ?? 
>
> 
>



## Generalized Advantage Estimation





## Deterministic Policy Gradient (DPG)

In stochastic policy gradient, the policy function $\pi(\cdot \mid s)$ is modeled as a probability distribution over actions. In contrast, the deterministic policy gradient method makes more bold decision with policy function that outputs deterministic action $a = \mu(s)$ . DPG has a great advantage over SPG in high-dimensional action spaces. In the stochastic case, the policy gradient integrates over both state and action spaces, whereas in the deterministic case it only integrates over the state space. As a result, computing the stochastic policy gradient may require more samples, especially if the action space has many dimensions. On the other hand, deterministic policy will limit the exploration of full state and action space, therefore DPG exploits the *off-policy actor-critic* algorithm, which choose actions according to a stochastic behavior policy (to ensure adequate **exploration**), but to learn about a deterministic target policy (**exploiting** the efficiency of the deterministic policy gradient).  

We will discuss the actor-critic algorithm in next post, here we only extend the stochastic policy gradient theorem to the deterministic policy gradient theorem. 

> Suppose that the MDP satisfies conditions that: $p\left(s^{\prime} \mid s, a\right), \nabla_{a} p\left(s^{\prime} \mid s, a\right), \mu_{\theta}(s), \nabla_{\theta} \mu_{\theta}(s), r(s, a), \nabla_{a} r(s, a), p_{1}(s)$  are continuous in all parameters and variables $s, a, s'$ and $x$, which imply that $\nabla_{\theta} \mu_{\theta}(s) \text { and } \nabla_{a} Q^{\mu}(s, a)$ exist and that the deterministic policy gradient exists. Then 
>
> 
> $$
> \nabla_{\theta} J\left(\mu_{\theta}\right) = \mathbb{E}_{s \sim d^{\mu}}\left[\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}\left.(s, a)\right|_{a=\mu_{\theta}(s)}\right]
> $$
> 
>
> Proof. The proof is provided in supplementary material in [1].




## Deep Deterministic Policy Gradient (DDPG)



## Normalized Advantage Function (NAF)



## Input Convex Neural Networks (ICNN)





## Reference

[1] [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf )

[2] [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf )

[3] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748v1.pdf )

[4] [Policy Gradient Methods for Reinforcement Learning with FunctionApproximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) 

[5] [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#dpg) 

[6] [Notes on the Generalized Advantage Estimation Paper](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/) 


