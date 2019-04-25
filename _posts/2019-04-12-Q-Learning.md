---
layout: post
tags: reinforcement-learning
author: Chunpai
---

This post will cover the foundation of value-based methods in reinforcement learning, such as MDP, Bellman equations, generalized policy iteration, MC, TD, and DQN. 

* TOC
{: toc}
### Notations

* $s, s':$ denotes states

* $a:$ an action

* $r:$ a reward

* $\mathcal{S}:$ set of nonterminal states

* $\mathcal{S}^+:$ set of all states, including the terminal states

* $\mathcal{A}:$ set of all actions or action space

* $\mathcal{R}:$ set of all possible rewards, a finite subset of $\mathbb{R}$

* $t:$ discrete time step

* $T:$ final time step of episode

* $A_t:$ action at time $t$

* $S_t:$ state at time $t$

* $R_t:$ reward at time $t$

* $\pi:​$ policy, or decision making rule

* $\pi(s):$ action taken in state $s$ under *deterministic* policy $\pi$

* $\pi(a \mid s):​$ probability of taking action $a​$ in $s​$ under *stochastic* policy $\pi​$

* $\pi(a\ \mid s, \theta):$ probability of taking action $a$ in $s$ given parameter $\theta$ 

* $p(s', r \mid s, a):$ probability of transition to state $s'$ with reward $r$, from state $s$ and action $a​$     

  

$$
p(s', r | s, a) = Pr(S_t = s', R_t=r |S_{t-1}=s, A_{t-1}=a) \\
$$

$$
\sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) = 1 \ \ \text{for all} \ s \in S, a \in A
$$

* $p(s' \mid s, a):​$ probability of transition to state $s'​$, from state $s​$ and action $a​$  


  $$
  p(s'| s, a) = Pr(S_t = s'|S_{t-1}=s, A_{t-1}=a) = \sum_{r\in \mathcal{R}} p(s', r | s, a) 
  $$

* $r(s, a, s'):​$ *expected* immediate reward on transition from $s​$ to $s'​$ under action $a​$ 


$$
r(s,a,s') = \mathbb{E}[R_t |S_{t-1}=s, A_{t-1} = a, S_{t}=s'] =\sum_{r\in \mathcal{R}} r \cdot p(r |s, a, s') =  \sum_{r\in \mathcal{R}} r \cdot \frac{p(s',r|s,a)}{p(s'|s,a)} 
$$


* $r(s, a):​$ *expected* reward for state-action pairs 


$$
r(s,a) = \mathbb{E}[R_t |S_{t-1}=s, A_{t-1} = a] =\sum_{r\in \mathcal{R}} r \sum_{s'\in S}p(s', r |s, a)
$$


* $G_t:$ return (cumulative reward) following time $t$, in the simplest case: 


$$
G_t = R_{t+1} + R_{t+2} + \cdots + R_{T}
$$




### Markov Decision Process

Markov decision processes formally describe an environment for reinforcement learning. Below is the figure about the process of agent-environment interaction in a Markov decision process.

![agent_environment_interaction](/assets/img/agent_environment_interaction.png)



The agent interacts with the environment over time and generate a sequence or trajectory:
$$
 \tau = \{ S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \cdots \}
$$


#### Markov Property

A state $S_t$ is Markov if and only if 


$$
\mathbb{P}\left[S_{t+1} | S_{t}\right]=\mathbb{P}\left[S_{t+1} | S_{1}, \ldots, S_{t}\right]
$$


that means the state is a sufficient statistic of the future. For a Markov state $s$ and successor state $s^{\prime}$, the state transition probability is defined by 


$$
\mathcal{P}_{s s^{\prime}}=\mathbb{P}\left[S_{t+1}=s^{\prime} | S_{t}=s\right]
$$


and we have state transition matrix $ \mathcal{P} $ as:


$$
\mathcal{P} = 
\left[\begin{matrix}
	\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
	\vdots & & \vdots \\
	\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
\end{matrix}\right]
$$


where each row of the matrix sums to $1$. 



#### Markov Process

A Markov Process (or Markov Chain) is a tuple $\langle\mathcal{S}, \mathcal{P}\rangle$ , 

* $\mathcal{S}$ is a (finite) set of states 
* $\mathcal{P}$ is a state transition probability matrix 

For example, below defines a Markov Chain of student activities in a day:

![Student Markov Chain](/assets/img/student_markov_chain.png)

where we can sample episodes with different states. 



#### Markov Reward Process

A Markov Process (or Markov Chain) is a tuple $\langle\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma \rangle​$ , 

- $\mathcal{S}​$ is a (finite) set of states 
- $\mathcal{P}$ is a state transition probability matrix 


$$
\mathcal{P}_{s s^{\prime}}=\mathbb{P}\left[S_{t+1}=s^{\prime} | S_{t}=s\right]
$$


- $\mathcal{R}$ is a reward function, $$ \mathcal{R}_{s}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s \right]$$  
- $\gamma​$ is a discount factor, $ \gamma \in[0,1]​$ , which is used to define the total discounted reward from time-step $t​$ 


$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots=\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}
$$


For example, below defines a Markov Reward Process of student activities in a day:

![Student MRP](/assets/img/student_mrp.png)

The discount factor is used to avoid infinite returns in infinite horizon, which will reduce the uncertainty about the future may not be fully represented. 



#### Markov Decision Process

A Markov decision process is a Markov reward process with decisions, which can be defined by a tuple $\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 

* $\mathcal{S}$ is a (finite) set of states 
* $\mathcal{A}​$ is a finite set of actions 
* $\mathcal{P}$ is a state transition probability matrix 


$$
\mathcal{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right]
$$


* $\mathcal{R}$ is a reward function, $$\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_t = a \right]$$ 
* $\gamma$ is a discount factor, $ \gamma \in[0,1] $ 

For example, below defines a Markov Decision Process of student activities in a day:

![Student MDP](/assets/img/student_mdp.png)

Another reason of using discounted factor is the decision $A_t$ typically has discounted effect on the future rewards $R_{t+k}$, which supposed to be effected by $A_{t+k-1}$. 



#### Ergodic MDP 

An ergodic Markov process has a limiting stationary distribution, which is a vector $d(\mathcal{S})$ about state distribution , such that

$$
d(\mathcal{S}) = d(\mathcal{S}) \cdot \mathcal{P}
$$


or 


$$
d(s)=\sum_{s^{\prime} \in \mathcal{S}} d\left(s^{\prime}\right) \mathcal{P}_{s^{\prime} s}
$$



It means over the long run, no matter what the starting state was, the proportion of time the chain spends in state $s$ is approximately $d(s)$, where $\sum_{s \in \mathcal{S} } d(s)$ = 1.0 .

An MDP is ergodic if the Markov chain induced by any policy is ergodic. For any policy $\pi$ , an ergodic MDP has an average reward per time-step $\rho^{\pi}$ that is independent of start state 


$$
\rho^{\pi}=\lim _{T \rightarrow \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=1}^{T} R_{t}\right]
$$






### Goal of an Agent

The goal of an agent is to maximize the expected value of the cumulative sum of a reward, which an agent receives in the long run.  In general, we seek to maximize the expected return, which is denoted by $\mathbb{E}[G_t]$. In episodic tasks, which means there exists finite steps of agent-environment interaction, the return can be generally expressed as


$$
G_1 = R_{1} + R_{2} + \cdots + R_{T} = R_{1} + G_{2}
$$
However, if the agent continually interacts with the environment without limit, which is called continuing tasks, we need another form of return 


$$
G_1 = R_{1} + \gamma R_{2} + \gamma^2 R_{3} + \cdots = \prod_{t=0}^{\infty} \gamma^{t} R_{t+1} = R_{1} + \gamma \cdot G_{2}
$$


The intermediate reward usually is designed by human, and keep in mind that, the reward signal is your way of communicating to the robot or agent what you want it to achieve, not how you want it achieved. If you want to give prior knowledge about how we want it to achieve, we might impact the initial policy or initial value function.



### Bellman Equations

**State-value Function:** the value of a state $s$ under policy $\pi$ is defined as


$$
v_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s\right]=\mathbb{E}_{\pi}\left[\prod_{k=0}^{\infty} \gamma^{k} R_{t+k+1} | S_{t}=s\right] \quad \forall s \in S
$$


which denotes by the expected return of state $s$ under policy $\pi$ . If $s$ is a terminated state, then $v_{\pi}(s) = 0$.  

**Action-value Function:** the value of a state-action pair action under policy $\pi$ is defined as 


$$
q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_{t} | S_t = s, A_t = a] = \mathbb{E}_{\pi} [\prod_{k=0}^{\infty} \gamma^k R_{t+k+1} |S_t =s , A_t=a] \ \ \ \  \forall s \in \mathcal{S},a \in \mathcal{A}
$$


which denotes by the expected return of state $s$ with action $a$ committed under policy $\pi$. 



#### Bellman Expectation Equation for $v_{\pi}(s)$


$$
\begin{aligned} v_{\pi}(s) &=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s\right]=\mathbb{E}_{\pi}\left[\prod_{k=0}^{\infty} \gamma^{k} R_{t+k+1} | S_{t}=s\right] \quad \forall s \in S \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) | S_{t}=s\right] \\ &=\sum_{a} \pi(a | s) \sum_{r, s^{\prime}} p\left(s^{\prime}, r | s, a\right)\left[r + \gamma v_{\pi}\left(s^{\prime}\right)\right] \\ &=\sum_{a} \pi(a | s) \cdot q_{\pi}(s, a) \end{aligned}
$$


The inner summation is to compute the expected return of state $s$ with a specific action $a$. The outer summation is to compute the expected return of state $s$ with all possible actions. We can use two backup diagrams to represent it:

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![bellman_exp_v_backup_diagram](/assets/img/bellman_exp_v_backup_diagram.png) | ![bellman_exp_v_backup_diagram2](/assets/img/bellman_exp_v_backup_diagram2.png) |





#### Bellman Expectation Equation for $ q_{\pi}(s, a)$ 


$$
\begin{aligned} q_{\pi}(s, a) &=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s, A_{t}=a\right]=\mathbb{E}_{\pi}\left[\prod_{k=0}^{\infty} \gamma^{k} R_{t+k+1} | S_{t}=s, A_{t}=a\right] \quad \forall s \in \mathcal{S}, a \in \mathcal{A} \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) | S_{t}=s, A_{t}=a\right] \\ &=\sum_{r, s^{\prime}} p\left(s^{\prime}, r | s, a\right)\left[r+ \gamma v_{\pi}\left(s^{\prime}\right)\right] \end{aligned}
$$


We can also use two backup diagrams to represent it:

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![bellman_exp_q_backup_diagram](/assets/img/bellman_exp_q_backup_diagram.png) | ![bellman_exp_q_backup_diagram](/assets/img/bellman_exp_q_backup_diagram2.png) |





#### Bellman Optimality Equation for $v_{\pi}(s)$ 

If we have already known the optimal policy where 


$$
\begin{aligned} v_{*}(s) &=\max _{\pi} v_{\pi}(s) \\ q_{*}(s, a) &=\max _{\pi} q_{\pi}(s, a) \end{aligned}
$$


we can get that 


$$
v_{*}(s)=\max _{a} q_{*}(s, a) \quad \forall s \in \mathcal{S}
$$


that means the value of a state under an optimal policy must equal the expected return for the best action from that state.


$$
\begin{align}
v_{*}(s) &= \max_{a} q_{*}(s,a) \\
&= \max_{a} \mathbb{E}_{\pi^*}[G_t | S_t = s, A_t = a] \\
&= \max_{a} \mathbb{E}_{\pi^*}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \max_{a} \mathbb{E}_{\pi^*}[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] \\
&= \max_{a}\sum_{r,s'} p(s', r |s, a) [r + \gamma v_{\pi^{*}}(s')] \\
&= \max_{a} \sum_{r,s'} p(s', r |s, a) [r + \gamma v_{\pi^{*}}(s')] \\
&= \pi^{*}(a|s) \sum_{r,s'} p(s', r |s, a) [r + \gamma v_{\pi^{*}}(s')]
\end{align}
$$


|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![bellman_opt_v_backup_diagram](/assets/img/bellman_opt_v_backup_diagram.png) | ![bellman_opt_v_backup_diagram2](/assets/img/bellman_opt_v_backup_diagram2.png) |



#### Bellman Optimality Equation for $q_{\pi}(s, a)$ 


$$
\begin{align}
q_{*}(s, a) &= \mathbb{E}_{\pi^*}[G_{t} | S_t = s, A_t = a] \\
    &= \mathbb{E}_{\pi^*}[R_{t} + \gamma G_{t+1} | S_t = s, A_t = a] \\
    &= \mathbb{E}_{\pi^*}[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] \\
    &= \mathbb{E}_{\pi^*}[R_{t+1} + \gamma \max_{a'} q_{*}(S_{t+1}, a') | S_t = s, A_t = a] \\
    &= \sum_{r,s'} p(s', r |s, a) [r + \gamma \max_{a'} q_{\pi^{*}}(s', a')]
    \end{align}
$$

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![bellman_opt_q_backup_diagram](/assets/img/bellman_opt_q_backup_diagram.png) | ![bellman_opt_q_backup_diagram2](/assets/img/bellman_opt_q_backup_diagram2.png) |





### Reference

[1] [David Silver's Lecture on MDP](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf) 

[2] 

[3]


