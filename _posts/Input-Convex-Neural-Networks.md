---
layout: post
tags: structured-prediction
author: Chunpai
---

This post is about the deep learning for structure prediction. This post is a study note on paper [Input Convex Neural Networks](https://arxiv.org/pdf/1609.07152.pdf) on structure prediction. 



* TOC
{: toc}
## Goal and Motivations

- Design of a deep neural network such that the output of the network is a convex function of (some of) the inputs

- The benefit of convexity is to optimize over the convex inputs $y$ to the network given some fixed value for other inputs $x$ :

  ​	
  $$
  \arg\min_{y} f(x, y, \theta)
  $$





- Many applications can be formulated as the objective function above, such as *structure prediction, data imputation, continuous action reinforcement learning* 





## Methods

This paper first introduce the fully input convexity for applications such as data imputation, however, it is not necessary to use fully convexity on other applications, such as structured prediction, which thus can use partially input convex neural network. For structure prediction, we can use neural network to build a joint model over an input $x$ and output $y$ example space, and only convexity over the output $y$. Once we build neural network to represent the the convex function $f(x,y,\theta)$ w.r.t $y$, we may easily find the global optimal. However, the authors state that in practice this problem still involves the solution of a potentially very complex optimization problem,  which may has high computation cost. (On the other hand, the inference problem can be formulated as a linear program. ) 



Two approximate inference methods can be applied: *1.) the gradient descent based inference; 2.) the bundle entropy method*. The gradient descent based method uses the updating rule: 

 
$$
\hat{y} \leftarrow \mathcal{P}_{\mathcal{Y}}\left(\hat{y}-\alpha \nabla_{y} f(x, \hat{y} ; \theta)\right)
$$


which is appealing simple, but have to use projected sub-gradient descent for non-smooth function. In addition, the performance of this method is not stable and requires additional hyper-parameters tuning. 



### Bundle Entropy Method

Bundle entropy method is an extension of bundle method [2], which is known as the epigraph cutting plane approach.  [3] is a good tutorial about cutting plane method. One trivial example of cutting plane method is bisection on $\mathbb{R}$. Cutting plane methods is to find constraints (cutting plane) based on first order condition that separating the optimal solution sets and  the current solution (query point) $x$. First, we need to check if current query solution is optimal or feasible, if not, then we can define a constraint or cut that optimal solution must satisfy based on current query. The difficult part of cutting plane method is how to choose the query points effectively to boost the convergence rate.

For epigraph cutting plane method, 

In this paper, the authors propose the bundle entropy method for this inference problem specifically. 

## Reference



[1] [Input Convex Neural Networks](https://arxiv.org/pdf/1609.07152.pdf)

[2] [Bundle methods for machine learning.](Le, Quoc V., Alex J. Smola, and Svn Vishwanathan. "Bundle methods for machine learning." *Advances in neural information processing systems*. 2008.)

[3] [Localization and Cutting-Plane Methods](https://see.stanford.edu/materials/lsocoee364b/05-localization_methods_notes.pdf)




