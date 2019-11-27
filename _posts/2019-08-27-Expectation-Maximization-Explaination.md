---
layout: post
tags: approximate-inference
author: Chunpai
---

I had written a note about the expectation maximization couple years ago. Recently, my research topic changes to the field of knowledge tracing, and many classic methods (i.e. BKT, SPARFA) leverage EM in their algorithm. I have to recap the EM algorithm and implement those baseline methods. In this post, I will go over some examples of using EM algorithm, such as *Gaussian Mixture Model (GMM)* and *Hidden Markov model (HMM)*. 

In order to understand the implementation throughly, it is better to read my note in the following link first: [Expectation Maximization](/assets/note/Expectation_Maximization_Explanation.pdf). Please leave comments if you find any issues or questions. Thanks.

* TOC
{: toc}
## Coin Tossing

Assume 3 coins A, B, C, and the probabilities of tossing head are unknown for these 3 coins, which are denoted as $\pi, p, q$ respectively. Now there is an experiment, we toss coin $A$ first. If $A$ is head, then toss $B$; if $A$ is tail, then toss $C$. We have 10 trails, and observed following results:

```
1,1,0,1,0,0,1,0,1,1
```

Now we would like to know if these results of coin $B$ or coin $C$, but we do not know the parameters $\pi, p, q$, we have to estimate them based on the observations. 



## Gaussian Mixture Model

## Hidden Markov Model

## Bayesian Knowledge Tracing

## Implementation of EM

```python
import sklearn 
import numpy as np
import scipy

a = np.zeros(10)
print("ths list {}".format(a))
```









