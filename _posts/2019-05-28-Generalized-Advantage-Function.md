---
layout: post
tags: reinforcement-learning
author: Chunpai
---

The generalized advantage estimation (GAE) summaries the different variants of actor-critic algorithms, which tries to find a good estimate of policy gradient. 

* TOC
{: toc}
## Generalized Advantage Estimator

Recall that, the (undiscounted) policy gradient is related to the empirical returns $R(\tau)$: 


$$
\begin{align}
\nabla_{\theta} J(\theta) 
&= \nabla_{\theta}\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[ R(\tau)\right]\\
&= \nabla_{\theta}\mathbb{E}\left[ \sum_{t=0}^{\infty} r(s_t, a_t) \right]  \\
&=\mathbb{E} \left[ \sum_{t=0}^{\infty} \left(\sum_{t=0}^{\infty} r(s_t, a_t) \right)  \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right] 
\end{align}
$$


however, the variance of the gradient estimate scales unfavorably with the time horizon, since the effect of an action is confounded with the effects of past and future actions. Actor-critic methods *use a value function rather than the empirical returns*, obtaining an estimator with significant lower variance at the cost of introducing a tolerable level of bias. 



There are several variants of the policy gradient above, which have the form 


$$
\nabla_{\theta} J(\theta) = \mathbb{E}\left[\sum_{t=0}^{\infty} \Psi_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right]
$$


where $\Psi_t$ may be one of the following:

1. $\sum_{t=0}^{\infty} r(s_t, a_t)$ : total reward of the trajectory
2. $\sum_{t'=t}^{\infty} r(s_{t'}, a_{t'})$: reward following action $a_t$ 
3. $\left(  \sum_{t'=t}^{\infty} r(s_{t'}, a_{t'}) \right) - b(s_t)$: reward following action $a_t$,  subtracting a baseline
4. $Q^{\pi}(s_t, a_t)$ : state-action value function
5. $A^{\pi}(s_t, a_t)$: advantage function which equals to $Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$ 
6. $r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$: TD residual

 

Among all those above, the choice $\Psi_t = A^{\pi}(s_t, a_t)$ yields almost the lowest possible variance, because the advantage function measures whether or not the action is better or worse than the policy's default behavior. 





## Off-Policy Actor-Critic





## Reference


[1] 

[2]

[3]


