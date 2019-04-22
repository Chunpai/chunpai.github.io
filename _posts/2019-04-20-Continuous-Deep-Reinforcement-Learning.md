---
layout: post
tags: reinforcement-learning
author: Chunpai
---

In this post, I will summarize three papers related to the deep reinforcement learning in continuous action space, which are 

- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf )

- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf )

- [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748v1.pdf )

  

### Goal of Reinforcement Learning

A policy $\pi$​ is a mapping from states to probabilities of selecting each possible action. At step $t$ of an episode,     


$$
\pi(a | s)=\operatorname{Pr}\left(A_{t}=a | S_{t}=s\right)
$$


We can view the policy $$\pi$$ as a neural network which parameterized by $$\theta$$, which are denoted by $ \pi(a \|s, \theta) $ or $\pi_{\theta} (a \| s)$.  The goal of reinforcement learning is to find a policy $$\pi$$ that achieve a lot reward over the long run. For example, in episodic tasks, the probability of occurrence of one specific episode $$\tau$$ under policy $\pi$ is 


$$
p_{\theta}(\tau)
=p_{\theta}\left(s_{1}, a_{1}, \cdots, s_{T}, a_{T}\right)
=p\left(s_{1}\right) \prod_{t}\left[\pi_{\theta}\left(a_{t} | s_{t}\right) \cdot p\left(s_{t+1} | s_{t}, a_{t}\right)\right]
$$


The objective is to find the optimal policy $$\pi$$ which is parameterized by $$\theta$$ such that have maximum expected return:


$$
\theta^{*}=\arg \max _{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t}^{T} r\left(s_{t}, a_{t}\right)\right]
$$


Once we obtain the optimal policy, we can determine the optimal action in each possible state. 

### Policy Gradient

For each trajectory $\tau$, there is a total reward $R$ associate with it, 


$$
R(\tau) = \sum_{t=1}^{T} r(s_t, a_t)
$$


Then the expected total reward of a trajectory under $\theta$ is 




$$
\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]
$$




We can maximize the $$\bar{R}_{\theta}$$ w.r.t  $\theta$ with gradient descent. We firstly compute the $\nabla_{\theta} \bar{R}_{\theta}$ 


$$
\begin{align}
\nabla \bar{R}_{\theta} &= \sum_{\tau} R(\tau) \nabla p_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)} \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau)\\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] \\
&\color{red}{\approx} \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \nabla \log p_{\theta}(\tau^n) \\
&= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log \left\{ p\left(s_{1}^{n} \right) \prod_{t}\left[\pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot p\left(s_{t+1}^{n} | s_{t}^{n}, a_{t}^{n}\right)\right]\right\} \\
&= \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \nabla \log \left[  \prod_{t}\pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \right] \\
&= \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \sum_{t} \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n} \right) \\
&=\frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} R(\tau^n) \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
\end{align}
$$


where we apply the fact in 3rd step that 


$$
\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}
$$


Now given a policy $\pi_{\theta}$ , we need to first collect many trajectories with current policy, then we compute $$\nabla \bar{R}_{\theta}$$ and apply gradient descent:


$$
\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}
$$


We can see that if current $\pi_{\theta} ( a_{t}^{n} \| s_{t}^{n} )$ leads to positive total reward, then the gradient update will increase the probability $\pi_{\theta} ( a_{t}^{n} \| s_{t}^{n} )$. If negative total reward, then it will decrease the probability.

#### Tip 1: Add a Baseline

Sometimes, the total rewards of all trajectories are always positive, then the gradient update will always increase the probability of all $\pi_{\theta} ( a_{t}^{n} \| s_{t}^{n} )$, since $$\nabla \bar{R}_{\theta}$$ is positive. Although the increasing magnitude are different among different actions, we need to ensure the $\sum_{j} \pi_{\theta} ( a_{tj}^{n} \| s_{t}^{n} ) = 1.0 $ with normalization, which may actually decrease the probability of good actions. Notice that, we are doing sampling to approximate the expectation, and at the worse case, some good actions may not be sampled ever, then the probabilities of all other (including bad) actions are increased. After normalization, it results in low probabilities of good actions which were not sampled. 

We may want to assign negative rewards to bad actions so that the gradient update will decrease the probability. We can modify the $$\nabla \bar{R}_{\theta}$$ as below:


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [R(\tau^n) - b] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


where $b$ is called baseline, which can be assign as 


$$
b \approx E[R(\tau)] 
$$


during we sample the trajectories. 



#### Tip 2: Assign Suitable Credit

From the formulation of gradient $$\nabla \bar{R}_{\theta}$$, the gradient $ \nabla \log \pi_{\theta}(a_{t}^{n} \| s_{t}^{n} )$  all actions in the same trajectory always multiply same value $$R(\tau^n) - b$$ . It would be more reasonable to assign high value to $ \pi_{\theta}(a_{t}^{n} \| s_{t}^{n} ) $ which action leads to high return in the following steps. Therefore, we modify the  $$\nabla \bar{R}_{\theta}$$ as below: 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [G_{t}^{n} - b] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


#### Tip 3: Actor-Critic

Again, $G_t^n$ is collected via sampled trajectories. For same state and action, the variance of $G_t^n$ may be very high if we do not have sufficient samples, which results in very unstable training. Therefore, we can replace the $G_t^n$ with $E[G_t^n]$ . Recall that $E[G_t^n]$ is just the Q-value $Q(s_t^n, a_t^n)$. We can also assign the value of baseline as $b = V(s_t^n)$, and we derive the so-called advantage function:


$$
A(s_t^n, a_t^n) = Q(s_t^n, a_t^n) - V(s_t^n)
$$


which basically describe the advantage of current action $a_t^n$ at state $s_t^n$ compared with the average performance of other actions relatively. Now, we have 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [Q_{t}^{n}(s_t^n, a_t^n) - V(s_t^n)] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


The most exciting thing here is we combine the value based method with policy-based method. Here policy-based method can be viewed as an actor that try to generate a good action, and value-based method can be viewed as a critic to evaluate how good is the action. If the critic says it is a good action, then the actor will increase the probability of this action; otherwise decrease. An actor adjusts the parameter $\theta$ of the stochastic policy $\pi_{\theta}(a \| s)$ by stochastic gradient ascent. A critic parameterized by $w$ estimates the action-value function $Q^{w}(s, a) \approx Q^{\pi}(s, a)$ using an appropriate policy evaluation algorithm such as temporal-difference learning. 



### Deterministic Policy Gradient 



  

