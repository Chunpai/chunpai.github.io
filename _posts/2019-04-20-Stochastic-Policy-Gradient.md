---
layout: post
tags: reinforcement-learning
author: Chunpai
---

In this post, I will summarize some concepts about policy gradient, off-policy gradient, and importance sampling. 



* TOC
{: toc}


### Goal of Reinforcement Learning

A policy $ \pi$​ is a mapping from states to probabilities of selecting each possible action. At step $t$ of an episode,     


$$
\pi(a | s)=\operatorname{Pr}\left(A_{t}=a | S_{t}=s\right)
$$


We can view the policy $$\pi$$ as a neural network which parameterized by $$\theta$$, which are denoted by $ \pi(a \mid s, \theta) $ or $\pi_{\theta} (a \mid  s)$.  The goal of reinforcement learning is to find a policy $$\pi$$ that achieve a lot reward over the long run. For example, in episodic tasks, the probability of occurrence of one specific episode $\tau$ under policy $\pi$ is 


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

First, we can do sampling and averaging to derive an unbiased estimate of our objective $J(\theta)​$ :


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
J(\theta) = \bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]
$$




We can maximize the $$ \bar{R}_{\theta}$$ w.r.t  $\theta$ with gradient descent. We firstly compute the $\nabla_{\theta} \bar{R}_{\theta}$ 


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

Now given a policy $ \pi_{\theta}​$ , we need to 1.) first collect many trajectories with current policy, 2.) accumulate or estimate the return, 3.)  compute $$\nabla \bar{R}_{\theta}​$$ and apply gradient descent [REINFORCE algo.]:



$$
\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}
$$



where $$ \nabla \bar{R}_{\theta}​$$ is *a stochastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument* $\theta​$ . 

![REINFORCE algorithm, a on-policy policy gradient method](/assets/img/REINFOCE_algo.png)



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
&= \frac{1}{N} \sum_{n=1}^{N}  \sum_{t=1}^{T_n} \left[ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot \color{red}{\hat{Q}_{t}^{n}} \right]
\end{align}
$$

where it is not unbiased estimator anymore. **Here we add some bias and reduce some variance ?**. Recall the definition of variance is MSE between estimated reward multiplier and true reward multiplier. Here, since we reduce the reward multiplicity from $\sum_{t=1}^{T_n} r(s_t, a_t)$ to $\sum_{t'=t}^{T_n} r(s_{t'}, a_{t'})$ , the estimated reward multiplier is reduced to a more accurate one, thus the variance of our estimated policy gradient reduced. Am I right ?

We can also understand in the following way: the original gradient $ \nabla \log \pi_{\theta}(a_{t}^{n} \mid  s_{t}^{n} )​$  of all actions in the same trajectory always multiply same value $R(\tau^n) ​$ , and intuitively it would be more reasonable to assign high value to $\pi_{\theta}(a_{t}^{n} \mid  s_{t}^{n} ) ​$ which action leads to high return in the following steps. 



#### Tip 1: Add a Baseline

Sometimes, the total rewards of all trajectories are always positive, then the gradient update will always increase the probability of all $\pi_{\theta} ( a_{t}^{n} \mid s_{t}^{n} )​$, since $$\nabla \bar{R}_{\theta}​$$  is positive. Although the increasing magnitude are different among different actions, we need to ensure the $\sum_{j} \pi_{\theta} ( a_{tj}^{n} \mid  s_{t}^{n} ) = 1.0 ​$ with normalization, which may actually decrease the probability of good actions. Notice that, we are doing sampling to approximate the expectation, and at the worse case, some good actions may not be sampled ever, then the probabilities of all other (including bad) actions are increased. After normalization, it results in low probabilities of good actions which were not sampled, and the good actions may not be sampled ever since then.

We may want to assign negative rewards to bad actions so that the gradient update will decrease the probability. We can modify the $ \nabla \bar{R}_{\theta}​$ as below:


$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t}^{T_n} \left\{ \nabla \log \pi_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right) \cdot [\hat{Q}_t^n - b] \right\}
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



where $g(\tau) = \nabla_{\theta} \log p_{\theta}(\tau)$. From above, we can derive the best baseline with minimum variance:



$$
b=\frac{E\left[g(\tau)^{2} r(\tau)\right]}{E\left[g(\tau)^{2}\right]} = \frac{E\left[(\nabla_{\theta} \log p_{\theta}(\tau))^{2} r(\tau)\right]}{E\left[ (\nabla_{\theta} \log p_{\theta}(\tau))^{2}\right]}
$$


which is just expected reward, but weighted by gradient magnitudes. The computation cost on gradient magnitudes offsets the benefit of reduced variance, so we typically use the average reward as baseline.



### Off-Policy Gradient

Policy gradient, for example REINFORCE algorithm, is an on-policy method. It is inefficient to iteratively update the model $ \pi_{\theta} $ and then generate new trajectories. Off-policy method is to train the policy $\pi_{\theta}$, called *target policy*, by using the sampled trajectories generated by another policy $ \pi_{\omega}​$, called *behavior policy*. In this way, we can reuse the sample trajectories. 

Off-policy method has different form of objective function:


$$
J(\theta) = E_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right] = E_{\tau \sim \color{red}{p_{\omega}(\tau)}}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right]
$$


which leverages the *important sampling* 


$$
\begin{aligned} E_{x \sim p(x)}[f(x)] &=\int p(x) f(x) d x \\ 
&=\int \frac{q(x)}{q(x)} p(x) f(x) d x \\ 
&=\int q(x) \frac{p(x)}{q(x)} f(x) d x \\ 
&=E_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right] 
\end{aligned}
$$


We can compute the ratio $ \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)}​$ 


$$
\begin{align}
\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)}
&=\frac{p_{\theta}\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right) p_{\theta}\left(\mathbf{s}_{t+1} | \mathbf{s}_{t}, \mathbf{a}_{t}\right)}{ p_{\omega}\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} \pi_{\omega}\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right) p_{\omega}\left(\mathbf{s}_{t+1} | \mathbf{s}_{t}, \mathbf{a}_{t}\right)}\\
&=\frac{ \prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)}{\prod_{t=1}^{T} \pi_{\omega}\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)}
\end{align}
$$

where we use the fact that different policies do not effect the transition probabilities of the environment, that is $$ p_{\theta}(\mathbf{s}_{1}) = p_{\omega}(\mathbf{s}_{1})$$.





#### Policy Gradient with Important Sampling

We can derive the policy gradient w.r.t to parameter $\theta$ as:


$$
\begin{align}
\nabla_{\theta} J(\theta) &= E_{\tau \sim p_{\omega}(\tau)} \left[ \frac{\nabla p_{\theta}(\tau) }{p_{\omega}(\tau)} R(\tau)\right] \\
&= E_{\tau \sim p_{\omega}(\tau)} \left[ \frac{p_{\theta}(\tau)  \nabla \log p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right] \\
&= E_{\tau \sim p_{\omega}(\tau)} \left[ \frac{p_{\theta}(\tau)  }{p_{\omega}(\tau)} \nabla \log p_{\theta}(\tau) R(\tau)\right] \\
&= E_{\tau \sim p_{\omega}(\tau)}\left[\left(\prod_{t=1}^{T} \frac{\pi_{\theta} \left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)}{\pi_{\omega}\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)}\right)\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta} \left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]
\end{align}
$$


if $\theta = \omega$ , then we can remove $\frac{p_{\theta}(\tau)  }{p_{\omega}(\tau)} $ at 3rd step, and $$\nabla_{\theta} J(\theta) = E_{\tau \sim p_{\omega}(\tau)} \left[ \nabla \log p_{\theta}(\tau) R(\tau)\right]$$.  We can also do more analysis on last equation and try to reduce the variance. But we omit it here [6, 7, 8]. 



#### Issue of Importance Sampling 

Using importance sampling, we can derive unbiased policy gradient, that is 


$$
\begin{align}
\nabla_{\theta} J(\theta) &=  \nabla_{\theta} E_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right] \\
&= \nabla_{\theta} E_{\tau \sim p_{\omega}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right] \\
&\approx \nabla_{\theta} \left[ \frac{1}{N} \sum_{n=1}^{N} p_{\theta}(\tau) R(\tau)\right]\\
&= \nabla_{\theta} \hat{E}_{\tau \sim p_{\omega}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right] \\
&= \nabla_{\theta} \hat{E}_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right]
\end{align}
$$


where $\hat{E} $ and $\hat{J}$ denotes by the estimates. The last equality holds when $p_{\omega}$ have support everywhere $p_{\theta}$ does:  


$$
p_{\theta}(\tau) > 0 \Rightarrow p_{\omega}(\tau) > 0
$$




The precision of these estimates depend on the variances of $R(\tau)$ and $\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)$ respectively. 


$$
\begin{align}
\operatorname{Var}\left[ \hat{E}_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right] \right] 
&= \frac{\operatorname{Var}_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right]}{N} \\
&= \frac{\operatorname{E}_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau)^2 \right] - \left(\operatorname{E}_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right]\right)^2}{N} \\
\operatorname{Var}\left[ \hat{E}_{\tau \sim p_{\omega}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right] \right]
&= \frac{\operatorname{Var}_{\tau \sim p_{\omega} (\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right]}{N}\\
&= \frac{\operatorname{E}_{\tau \sim p_{\omega}(\tau)} \left[ \left( \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right)^2 \right] - \left( \operatorname{E}_{\tau \sim p_{\omega}(\tau)} \left[ \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau) \right]\right)^2}{N} \\
&= \frac{\operatorname{E}_{\tau \sim p_{\omega}(\tau)} \left[ \left( \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)}\right)^2 R(\tau)^2 \right] - \left(\operatorname{E}_{\tau \sim \color{red}{p_{\theta}(\tau)}} \left[R(\tau) \right]\right)^2}{N} \\
&= \frac{\operatorname{E}_{\tau \sim \color{red}{p_{\theta}(\tau)}} \left[ \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)^2 \right] - \left(\operatorname{E}_{\tau \sim \color{red}{p_{\theta}(\tau)}} \left[R(\tau) \right]\right)^2}{N}
\end{align}
$$


If the distribution $p_{\theta}(\tau)$ is very different from the $p_{\omega}(\tau)​$ , the precision of two different estimates would be very different. It is worth noting that importance sampling provides a way for variance reduction [4, 5] by restricting 


$$
\operatorname{Var}_{\tau \sim p_{\theta}(\tau)} \left[R(\tau) \right] - \operatorname{Var}_{\tau \sim p_{\omega} (\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)\right] > 0
$$


that is 


$$
\begin{align}
\operatorname{E}_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau)^2 \right] - \operatorname{E}_{\tau \sim p_{\theta}(\tau)} \left[ \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} R(\tau)^2 \right] &> 0 \\
\operatorname{E}_{\tau \sim p_{\theta}(\tau)} \left[ \left( 1- \frac{p_{\theta}(\tau)}{p_{\omega}(\tau)} \right)R(\tau)^2 \right] &> 0
\end{align}
$$


### Summary

- The policy gradient has high variance, and the gradient will be very noisy. 
- We can reduce the variance by sampling more trajectories or using much larger batches. It would be very difficult to tweak learning rate, and adaptive step size rules like ADAM could be helpful, but not perfect. 
- The key point of improving the policy gradient methods is to reduce the variance of the policy gradient. 



### Reference

[1] Chapter 13: Policy Gradient Methods, [Reinforcement Learning:
An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

[2] Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. (1999). [Policy gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf). NeurIPS. 

[3] [Lecture of Berkeley DRL course](https://www.youtube.com/watch?v=XGmd3wcyDg8&list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37&index=21)

[4] [Importance Sampling for Reinforcement Learning](https://timvieira.github.io/blog/post/2014/12/21/importance-sampling/)

[5] [Variance Reduction with Importance Sampling](https://web.archive.org/web/20170401030417/http://www.columbia.edu/~mh2078/MCS04/MCS_var_red2.pdf)

[6] Jie, Tang, and Pieter Abbeel. "[On a connection between importance sampling and the likelihood ratio policy gradient.](http://rll.berkeley.edu/~jietang/pubs/nips10_Tang.pdf)" NeurIPS. 2010.]

[7] Levine, Sergey, and Vladlen Koltun. "[Guided policy search.](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)" *International Conference on Machine Learning*. 2013.

[8] Schulman, John, et al. "[Proximal policy optimization algorithms](https://arxiv.org/pdf/1707.06347.pdf)." *arXiv preprint arXiv:1707.06347* (2017).



