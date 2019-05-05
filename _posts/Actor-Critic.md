---
layout: post
tags: reinforcement-learning
author: Chunpai
---





* TOC
{: toc}


### Variance on Policy Gradient



## Actor-Critic

Again, $G_t^n$ is collected via sampled trajectories. For same state and action, the variance of $G_t^n$ may be very high if we do not have sufficient samples, which results in very unstable training. Therefore, we can replace the $G_t^n$ with $E[G_t^n]$ . Recall that $E[G_t^n]$ is just the Q-value $Q(s_t^n, a_t^n)$. We can also assign the value of baseline as $b = V(s_t^n)$, and we derive the so-called advantage function:


$$
A(s_t^n, a_t^n) = Q(s_t^n, a_t^n) - V(s_t^n)
$$


which basically describe the advantage of current action $a_t^n$ at state $s_t^n$ compared with the average performance of other actions relatively. Now, we have 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [Q_{t}^{n}(s_t^n, a_t^n) - V(s_t^n)] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


The most exciting thing here is we combine the value based method with policy-based method. Here policy-based method can be viewed as an actor that try to generate a good action, and value-based method can be viewed as a critic to evaluate how good is the action. If the critic says it is a good action, then the actor will increase the probability of this action; otherwise decrease. An actor adjusts the parameter $\theta$ of the stochastic policy $\pi_{\theta}(a \mid  s)$ by stochastic gradient ascent. A critic parameterized by $w$ estimates the action-value function $Q^{w}(s, a) \approx Q^{\pi}(s, a)$ using an appropriate policy evaluation algorithm such as temporal-difference learning. 



### Batch and Online Actor-Critic Algorithm



### Discount Factors for Policy Gradients



## Critics as State-Dependent Baseline



## Control Variates: Action-Dependent Baseline: Q-Prop







### Reference



