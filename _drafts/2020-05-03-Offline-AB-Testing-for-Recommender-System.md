---
title: "Offline A/B Testing for Recommender System"
layout: post
tags: [offline-evaluation, causal-inference, recommender-system, reinforcement-learning]
author: Chunpai
---

This is a study note of paper "Offline A/B Testing for Recommender System" by Alexandre et. al. This paper proposes two variants of capped importance sampling that achieve low bias under much more realistic conditions. 

* TOC
{: toc}




## Problem Setting   

Given a display $x$ represented by a set of contextual features as well as a set of eligible products, the recommender system outputs a probability distribution $$\pi(A\mid X)$$ where $a$ is a top-k ranking on the eligible products and $k$ is the number of items that will be displayed to customers. Taking action $a$ in state $x$ generates a reward $$r \in [0, r_{\max}]$$ that could be interpreted as a click or a purchase, which could be specified by the system designer. 

## Online A/B Testing   

In online A/B tests, the objective is to compare performance of the current existing system *prod* (or logged policy, denoted by $\pi_p$) and the new invented system *test* (or target policy, denoted by $\pi_p$)  under the so-called *isolation assumption*, where a set of $n$ units $x$ are randomly assigned to either *prod* and *test*. We aim to measure the average difference of value $$\Delta  \mathcal{R}$$, based on the reward signal $$r \in [0, r_{\max}]$$ ,  


$$
\Delta \mathcal{R}\left(\pi_{p}, \pi_{t}\right)=\mathbb{E}_{\pi_{t}}[R]-\mathbb{E}_{\pi_{p}}[R]
$$


which is called *average treatment effect* and  


$$
\mathbb{E}_{\pi_{p}}[R]=\mathbb{E}[R | A] \pi_{p}(A | X) \mathbb{P}(X)
$$

$\Delta \mathcal{R}$ could be estimated by Monte-Carlo using the two datasets collected during the test  $$ \mathcal{S}_{p} = \{(x_i, a_i, r_i): i \in \mathcal{P}_{p} \} $$ and  $$ \mathcal{S}_{t} = \{(x_i, a_i, r_i): i \in \mathcal{P}_t \} $$ . The empirical estimator $$ \Delta \hat{\mathcal{R}} $$ is:  



$$
\Delta \hat{\mathcal{R}}\left(\pi_{p}, \pi_{t}\right)=\hat{\mathcal{R}}\left(\mathcal{S}_{t}\right)-\hat{\mathcal{R}}\left(\mathcal{S}_{p}\right)
$$



where $$\hat{\mathcal{R}}(\mathcal{S})$$ is the empirical average of rewards over $\mathcal{S}$ gathered during the online A/B test.



## What is Offline A/B Testing ?  

In offline setting, we do not randomly assign units to different systems. Instead we have only one set of $n$ historical i.i.d. samples $$ \mathcal{S}_n = \{ (x_i, a_i, r_i): i \in [n] \} $$ collected using the current production recommender system $\pi_{p}$. Therefore, we can directly estimate $$\mathbb{E}_{\pi_{p}}[R]$$ using $$\hat{\mathcal{R}}(\mathcal{S}_n)$$, but for $$\mathbb{E}_{\pi_{t}}[R]$$ we cannot use a direct estimation since we do not have data gathered under $$\pi_t$$ yet. However, this expected reward $$\mathbb{E}_{\pi_{t}}[R]$$ could be estimated by importance sampling using rewards gathered under the logging policy $$\pi_{p} $$. In the following sections,  we will discuss the limitations of different variants of estimators on $$\mathbb{E}_{\pi_{t}}[R]$$ . 

## Basic Importance Sampling   



## Capped Importance Sampling   




















