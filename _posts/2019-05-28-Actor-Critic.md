---
layout: post
tags: reinforcement-learning
author: Chunpai
---





* TOC
{: toc}
## Variance on Policy Gradient

We introduce more accurate reward multiplier in policy gradient to reduce the variance, and adding the baseline to get reasonable training and not change the bias. The goal of actor-critic is to reduce the variance, meanwhile, try to keep the bias stable.



## Actor-Critic

In policy gradient, $G_t^n = \sum_{t}^{T_n} r(s_t, a_t)$ is collected via sampled trajectories. For same state and action, the variance of $G_t^n$ may be very high if we do not have sufficient samples, which results in very unstable training. Therefore, we can replace the $G_t^n$ with $E[G_t^n]$ . Recall that $E[G_t^n]$ is just the Q-value $Q(s_t^n, a_t^n)$. We can also assign the value of baseline as $b = V(s_t^n)$, and we derive the so-called advantage function:


$$
A(s_t^n, a_t^n) = Q(s_t^n, a_t^n) - V(s_t^n)
$$


which basically describe the advantage of current action $a_t^n$ at state $s_t^n$ compared with the average performance of other actions relatively. Now, we have 


$$
\nabla J_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [Q_{t}^{n}(s_t^n, a_t^n) - V(s_t^n)] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


The most exciting thing here is we combine the value based method with policy-based method. Here policy-based method can be viewed as an actor that try to generate a good action, and value-based method can be viewed as a critic to evaluate how good is the action. If the critic says it is a good action, then the actor will increase the probability of this action; otherwise decrease. An actor adjusts the parameter $\theta$ of the stochastic policy $\pi_{\theta}(a \mid  s)$ by stochastic gradient ascent. A critic parameterized by $w$ estimates the action-value function $Q^{w}(s, a) \approx Q^{\pi}(s, a)$ using an appropriate policy evaluation algorithm such as temporal-difference learning. 



However, in order to compute the updated policy gradient, we still need to use two networks to compute the  $Q^{\pi}(s_t^n, a_t^n)$ and $V^{\pi}(s_t)$.  Note that 


$$
Q^{\pi}(s_t^n, a_t^n) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)}\left[ V^{\pi}(s_{t+1}) \right]
$$




If we accept a small approximation of $Q^{\pi}(s_t^n, a_t^n)$ with 


$$
Q^{\pi}(s_t^n, a_t^n) \approx r(s_t, a_t) + V^{\pi}(s_{t+1})
$$


by increasing a little bias but reducing a little variance, then we can have advantage function approximated as 


$$
A^{\pi}(s_t^n, a_t^n) \approx r(s_t, a_t) + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)
$$


Most of actor-critic algorithms fit the value function $V^{\pi}(s)$, but it is not the only solution, and we can also fit the Q-function when it is off-policy actor-critic with some advantages.



### Fit Value Function (Policy Evaluation)

Based on the generalized policy iteration, we know that we first need to evaluate the policy by computing the $V^{\pi}(s)$ . Rather than leveraging the Monte Carlo or Temporal Difference learning to perform the policy evaluation, which samples a lot trajectories to get a good approximation, we use a neural network as function approximator. 



![](/assets/img/value_function_approx.png)



The function approximator may not be as good as the general sampling method 


$$
V^{\pi}\left(\mathbf{s}_{t}\right) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)
$$


but it avoids the requirement of a lot sampling to get state-values for all states, and it can still get pretty good performance, by assuming that two slightly different states will get similar state-values, which leads to implicit variance reduction.



We can train the value function as the supervised learning, where training data is a set of state and state-value from some sampled trajectories with current policy $\pi$:



$$
\left\{\left(\mathbf{s}_{i, t}, \sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{i, t^{\prime}}, \mathbf{a}_{i, t^{\prime}}\right)  \right) \right\}
$$



where denote $y_{i,t} = \sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{i, t^{\prime}}, \mathbf{a}_{i, t^{\prime}}\right)  $ as the Monte Carlo target, and apply the supervised regression to minimize the mean square loss:



$$
\mathcal{L}(\phi)=\frac{1}{2} \sum_{i}\left\|\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i}\right)-y_{i}\right\|^{2}
$$



However, we use the sampled MC target, and the ideal target should be

 

$$
\begin{align}
y_{i, t}=\sum_{t^{\prime}=t}^{T} E_{\pi_{\theta}}\left[ r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right) \mid \mathbf{s}_{i, t} \right] &\approx r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\sum_{t^{\prime}=t+1}^{T} E_{\pi_{\theta}}\left[r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right) | \mathbf{s}_{i, t+1}\right] \\
& \approx r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right) + V^{\pi}(\mathbf{s}_{i, t+1}) \\
& \approx r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right) + \hat{V}^{\pi}_{\phi}(\mathbf{s}_{i, t+1})
\end{align}
$$



Note that, util we have fitted the value function, we do not have a good estimate of $V^{\pi}(\mathbf{s}_{i, t+1})$ . Instead, what we can do is to use the previous fitted value function $\hat{V}^{\pi}$ which is slightly incorrect value function. One advantage of doing this is in practice this incorrect value function has a lower variance target, which is better than a unbiased target with high variance. Therefore, we can obtain better training data as 



$$
\left\{\left(\mathbf{s}_{i, t}, r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t+1}\right)\right)\right\}
$$



and this kind of supervised training is known as *bootstrapped estimate*. What we can learn from here is, we would rather use the biased but low noised samples than the unbiased but high variance samples as training data, when the size of sampled training data is small. 

### Batch Actor-Critic Algorithm

The general actor-critic algorithm can be understood as the figure below: 



![](/assets/img/actor-critic.png)



and the batch mode of actor-critic algorithm 



![](/assets/img/batch_actor_critic.png)



where we can also (red line) use the bootstrapped estimate on reward sum to generate training target to introduce some bias but reduce some variance. 



### Discount Factors for Policy Gradients

It is better to get rewards sooner than later, and we use the discounted estimated target with $\gamma \in [0, 1]$:


$$
y_{i, t} \approx r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right) + \gamma \hat{V}^{\pi}_{\phi}(\mathbf{s}_{i, t+1})
$$


and policy gradient with critic and discount factor as: 


$$
{\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} | \mathbf{s}_{i, t}\right)\left(r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\gamma \hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t+1}\right)-\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t}\right)\right)}
$$


Thus, we have the batch actor-critic algorithms with discount as

![](/assets/img/batch_discount.png)



### Online Actor-Critic Algorithm

Instead of simulating the full trajectory with $\pi$, the online mode only needs to take the action at one step. 



 ![](/assets/img/online_discount.png)



One benefit of online mode is we can simulate the action to get $(s, a, s', r)$ and compute policy gradient in parallel to speed up the computation. 



![](/assets/img/parallel_actor_critic.png)







## Off-Policy Actor Critic



## Reference

[1] [Berkeley Reinforcement Learning Course on Actor-Critic](https://www.youtube.com/watch?v=Tol_jw5hWnI&list=PLLiwQX_Zp55SViaiVo2qzH5SqClB41AgO&index=11&t=2739s) 



