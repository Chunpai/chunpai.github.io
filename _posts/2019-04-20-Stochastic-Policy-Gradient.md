---
layout: post
tags: reinforcement-learning
author: Chunpai
---

In this post, I will summarize some concepts about policy gradient. 



### Goal of Reinforcement Learning

A policy $ \pi$​ is a mapping from states to probabilities of selecting each possible action. At step $t$ of an episode,     


$$
\pi(a | s)=\operatorname{Pr}\left(A_{t}=a | S_{t}=s\right)
$$


We can view the policy $$\pi​$$ as a neural network which parameterized by $$\theta​$$, which are denoted by $ \pi(a \mid s, \theta) ​$ or $\pi_{\theta} (a \mid  s)​$.  The goal of reinforcement learning is to find a policy $$\pi​$$ that achieve a lot reward over the long run. For example, in episodic tasks, the probability of occurrence of one specific episode $\tau​$ under policy $\pi​$ is 


$$
p_{\theta}(\tau)
=p_{\theta}\left(s_{1}, a_{1}, \cdots, s_{T}, a_{T}\right)
=p\left(s_{1}\right) \prod_{t}\left[\pi_{\theta}\left(a_{t} | s_{t}\right) \cdot p\left(s_{t+1} | s_{t}, a_{t}\right)\right]
$$


The objective is to find the optimal policy $\pi​$ which is parameterized by $$\theta​$$ such that have maximum expected return:


$$
\begin{align}
\theta^{*} &= \arg\max_{\theta} J(\theta) \\
&= \arg \max _{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t}^{T} r\left(s_{t}, a_{t}\right)\right] 
\end{align}
$$

First, we can do sampling and averaging to derive an unbiased estimate of our objective $J(\theta)$ :


$$
J(\theta) \approx \frac{1}{N} \sum_{n}^{N} \sum_{t}^{T}r\left(s_{t}, a_{t} \right)
$$


Then, we try to improve it based on gradient ascent. Once we obtain the optimal policy, we can determine the optimal action in each possible state. 

### Policy Gradient

For each trajectory $ \tau$, there is a total reward $R$ associate with it, 


$$
R(\tau) = \sum_{t=1}^{T} r(s_t, a_t)
$$


Then the expected total reward of a trajectory under $ \theta​$ is 




$$
\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]
$$




We can maximize the $$ \bar{R}_{\theta}​$$ w.r.t  $\theta​$ with gradient descent. We firstly compute the $\nabla_{\theta} \bar{R}_{\theta}​$ 


$$
\begin{align}
\nabla_{\theta} J(\theta) = \nabla \bar{R}_{\theta} &= \sum_{\tau} R(\tau) \nabla p_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)} \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau)\\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] \\
&\color{red}{\approx} \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \nabla \log p_{\theta}(\tau^n) \\
&= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log \left\{ p\left(s_{1}^{n} \right) \prod_{t}\left[\pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot p\left(s_{t+1}^{n} | s_{t}^{n}, a_{t}^{n}\right)\right]\right\} \\
&= \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \nabla \log \left[  \prod_{t}\pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \right] \\
&= \frac{1}{N} \sum_{n=1}^{N}  R(\tau^n) \sum_{t} \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n} \right) \\
&=\frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} R(\tau^n) \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \\
&=\frac{1}{N} \sum_{n=1}^{N} \left[ \sum_{t}^{T_n} r(s_t, a_t) \right] \left[\sum_{t}^{T_n} \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \right]
\end{align}
$$


where we apply the fact in 3rd step that 


$$
\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}
$$

The prove above is related to the *policy gradient theorem* [1], which provides us an analytic expression for the gradient of performance w.r.t the policy parameter that *does not* involve the derivative of the state distribution (model dynamic). 

Now given a policy $ \pi_{\theta}$ , we need to 1.) first collect many trajectories with current policy, 2.) accumulate or estimate the return, 3.)  compute $$\nabla \bar{R}_{\theta}​$$ and apply gradient descent [REINFORCE algo.]:



$$
\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}
$$



where $$ \nabla \bar{R}_{\theta}$$ is *a stochastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument* $\theta$ . 

#### Relationship between Policy Gradient and MLE

If we try to maximize the log-likelihood of a trajectory of $T$ state-action pairs $(s_t, a_t)​$ with gradient ascent, then we need to compute the gradient of log-likelihood as


$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{n}^{N} \left[ \sum_{t}^{T_n} \nabla_{\theta} \log \pi_{\theta}(a_t^n \mid s_t^n) \right]
$$


It is the $$ \nabla \bar{R}_{\theta}$$ without multiplying the return term $ \sum_{t}^{T_n} r(s_t, a_t) $. We can see that policy gradient ascent update is a formalization of trial-and-error: if current $\pi_{\theta} ( a_{t}^{n} \mid s_{t}^{n} )$ leads to positive total reward, then the gradient update will increase the probability $\pi_{\theta} ( a_{t}^{n} \mid  s_{t}^{n} )$. If negative total reward, then it will decrease the probability.



### Issues of Policy Gradient

#### Tip 0: Reducing Variance

Recall that, 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N}  \left[\sum_{t=1}^{T_n} \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \right] \left[ \sum_{t=1}^{T_n} r(s_t, a_t) \right]
$$


Simply use the distribution law, we have 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N}  \sum_{t=1}^{T_n} \left[ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \sum_{t=1}^{T_n} r(s_t, a_t) \right]
$$


Since the policy at time $t'$ cannot affect reward at time $t$ when $t < t'$ , we can revise the policy gradient as 


$$
\begin{align}
\nabla \bar{R}_{\theta} &\approx \frac{1}{N} \sum_{n=1}^{N}  \sum_{t=1}^{T_n} \left[ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \sum_{\color{red}{t'=t}}^{T_n} r(s_{t'}, a_{t'}) \right] \\
&= \frac{1}{N} \sum_{n=1}^{N}  \sum_{t=1}^{T_n} \left[ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot \color{red}{Q_{t}^{n}} \right]
\end{align}
$$




Recall the definition of variance is MSE between estimated reward multiplier and expected reward multiplier. Here, since we reduce the reward multiplicity from $\sum_{t=1}^{T_n} r(s_t, a_t)$ to $\sum_{t'=t}^{T_n} r(s_{t'}, a_{t'})$ , the estimated reward multiplier is reduced, thus variance reduced.

We can also understand in the following way: the original gradient $ \nabla \log \pi_{\theta}(a_{t}^{n} \mid  s_{t}^{n} )$  of all actions in the same trajectory always multiply same value $R(\tau^n) $ , and intuitively it would be more reasonable to assign high value to $\pi_{\theta}(a_{t}^{n} \mid  s_{t}^{n} ) ​$ which action leads to high return in the following steps. 



#### Tip 1: Add a Baseline

Sometimes, the total rewards of all trajectories are always positive, then the gradient update will always increase the probability of all $\pi_{\theta} ( a_{t}^{n} \mid s_{t}^{n} )​$, since $$\nabla \bar{R}_{\theta}​$$  is positive. Although the increasing magnitude are different among different actions, we need to ensure the $\sum_{j} \pi_{\theta} ( a_{tj}^{n} \mid  s_{t}^{n} ) = 1.0 ​$ with normalization, which may actually decrease the probability of good actions. Notice that, we are doing sampling to approximate the expectation, and at the worse case, some good actions may not be sampled ever, then the probabilities of all other (including bad) actions are increased. After normalization, it results in low probabilities of good actions which were not sampled, and the good actions may not be sampled ever since then.

We may want to assign negative rewards to bad actions so that the gradient update will decrease the probability. We can modify the $ \nabla \bar{R}_{\theta}$ as below:


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} \left\{ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot [Q_t^n - b] \right\}
$$


where $b$ is called baseline, which could be assign as average total reward practically


$$
b \approx E[R(\tau)] = \frac{1}{N} \sum_{n=1}^{N} R(\tau)
$$

during we sample the trajectories. But, are we allowed to do that ? We will see that subtracting a baseline is *unbiased in expectation*, that is 


$$
\begin{align}
\nabla \bar{R}_{\theta} &= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ \left( R(\tau)-b \right) \nabla \log p_{\theta}(\tau) \right] \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - E_{\tau \sim p_{\theta}(\tau)} \left[ b \nabla \log p_{\theta}(\tau) \right] \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - \int_{\tau} p_{\theta}(\tau)\cdot b \nabla \log p_{\theta}(\tau) d\tau \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - \int_{\tau}  b \nabla \cdot p_{\theta}(\tau) d\tau \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - b \nabla_{\theta}  \int_{\tau} p_{\theta}(\tau) d\tau \\
&=E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - b \nabla_{\theta} 1 \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \nabla \log p_{\theta}(\tau) \right] - 0
\end{align}
$$


In addition, the average reward baseline is not the best baseline, and we can derive the optimal baseline by analyzing the variance of policy gradient as below:


$$
\begin{align}
\operatorname{Var}[x] &= E\left[x^{2}\right]-E[x]^{2} \\
\nabla_{\theta} J(\theta) &= E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)(R(\tau)-b)\right] \\
\mathrm{Var[\nabla_{\theta} J(\theta)]} &= E_{\tau \sim p_{\theta}(\tau)}\left[\left(\nabla_{\theta} \log p_{\theta}(\tau)(R(\tau)-b)\right)^{2}\right]- \left\{E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)(R(\tau)-b)\right]\right\}^{2} \\
&= E_{\tau \sim p_{\theta}(\tau)}\left[\left(\nabla_{\theta} \log p_{\theta}(\tau)(R(\tau)-b)\right)^{2}\right]- \left\{ E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)R(\tau)\right]\right\}^{2} \\
\frac{d \mathrm{Var}}{d b}&=\frac{d}{d b} E\left[g(\tau)^{2}(r(\tau)-b)^{2}\right] \\
&=\frac{d}{d b}\left(E\left[g(\tau)^{2} r(\tau)^{2}\right]-2 E\left[g(\tau)^{2} r(\tau) b\right]+b^{2} E\left[g(\tau)^{2}\right]\right)\\
&= -2 E\left[g(\tau)^{2} r(\tau)\right]+2 b E\left[g(\tau)^{2}\right] \\
&=0
\end{align}
$$


where $g(\tau) = \nabla_{\theta} \log p_{\theta}(\tau)​$. From above, we can derive the best baseline with minimum variance:


$$
b=\frac{E\left[g(\tau)^{2} r(\tau)\right]}{E\left[g(\tau)^{2}\right]} = \frac{E\left[(\nabla_{\theta} \log p_{\theta}(\tau))^{2} r(\tau)\right]}{E\left[ (\nabla_{\theta} \log p_{\theta}(\tau))^{2}\right]}
$$
which is just expected reward, but weighted by gradient magnitudes. The computation cost on gradient magnitudes offsets the benefit of reduced variance, so we typically use the average reward as baseline.



### Off-Policy Gradient




### Actor-Critic

Again, $G_t^n​$ is collected via sampled trajectories. For same state and action, the variance of $G_t^n​$ may be very high if we do not have sufficient samples, which results in very unstable training. Therefore, we can replace the $G_t^n​$ with $E[G_t^n]​$ . Recall that $E[G_t^n]​$ is just the Q-value $Q(s_t^n, a_t^n)​$. We can also assign the value of baseline as $b = V(s_t^n)​$, and we derive the so-called advantage function:


$$
A(s_t^n, a_t^n) = Q(s_t^n, a_t^n) - V(s_t^n)
$$


which basically describe the advantage of current action $a_t^n$ at state $s_t^n​$ compared with the average performance of other actions relatively. Now, we have 


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} [Q_{t}^{n}(s_t^n, a_t^n) - V(s_t^n)] \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)
$$


The most exciting thing here is we combine the value based method with policy-based method. Here policy-based method can be viewed as an actor that try to generate a good action, and value-based method can be viewed as a critic to evaluate how good is the action. If the critic says it is a good action, then the actor will increase the probability of this action; otherwise decrease. An actor adjusts the parameter $\theta​$ of the stochastic policy $\pi_{\theta}(a \mid  s)​$ by stochastic gradient ascent. A critic parameterized by $w​$ estimates the action-value function $Q^{w}(s, a) \approx Q^{\pi}(s, a)​$ using an appropriate policy evaluation algorithm such as temporal-difference learning. 







### Reference

[1] Chapter 13: Policy Gradient Methods, [Reinforcement Learning:
An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

[2] Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. In Neural Information Processing Systems 12, pages 1057–1063.

[3] [Lecture of Berkeley DRL course](https://www.youtube.com/watch?v=XGmd3wcyDg8&list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37&index=21)



