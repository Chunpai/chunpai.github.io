---
layout: post
tags: recommender-system
author: Chunpai
---

* TOC
{: toc}
## Reviews of Recommender System

Recommendation models are usually classified into 3 categories: 

- collaborative filtering, which makes recommendations by learning from user-item historical **interactions**, either explicit (ratings, likes) or implicit feedback (e.g. browsing history). 
- content based recommendation, which is based primarily on similarities of different items and users with items' attributes and auxiliary information, such as texts, images, and videos.
- hybrid, which integrates two or more types of recommendation strategies.



**Implicit and Explicit Feedbacks:**

Implicit feedback indirectly reflects users' preference through behaviors like watching videos, purchasing products and clicking items. Compared to explicit feedback (i.e. rating and reviews), implicit feedback can be tracked automatically and is thus much easier to collect for content providers. However, implicit feedback is more challenge to utilize and interpret, since user satisfaction is not observed directly. 



**Pointwise Loss and Pairwise Loss**

The recommendation problem is to estimate the score of unobserved item for users. Pointwise learning is to minimize the squared loss between observed $y_{ui}$ and predicted $\hat{y}_{ui}$. If the target values are binarized, we may use the binary cross entropy loss function. Pairwise learning adopts the idea that observed entries should be ranked higher than the unobserved ones. As such, instead of minimizing the margin between observed entry $y_{ui}$ and $\hat{y}_{ui}$,  pairwise learning ~~also~~ maximizes the margin between observed entry $\hat{y}_{ui}$ and unobserved entry $\hat{y}_{uj}$. Pairwise learning is more difficult to train, and the performance is not stable. However, we may combine the pointwise the pairwise losses. 



**Sequential Recommendation**

The task of sequential recommendation can be considered as to generate a personalized item ranking list based on different types of user behavior sequences, which can be formally defined as:


$$
(p^1_{t+1}, p^2_{t+1}, \cdots, p^N_{t+1}) = f(b_1, b_2, \cdots, b_t, u)
$$


where sequence $\{b_1, \cdots, b_t\}$ represents the input sequence, $u$ represents the corresponding user of the sequence, and $p^i_{t+1}$ represents the probability that item $i$ will be liked by user $u$ at time $t+1$ .  



**Drawbacks of MF-based Methods:**

1.  a user or item's latent vector needs to be updated when the user have new experience, and computing cost is growing exponentially as the increase of the matrix size

2. they generally ignore the time dependency among behaviors both within a session and across different sessions.

   



**Tasks:** 

1. NCF implementations in TensorFlow and PyTorch, and check the TensorRec framework.
2.  Check the paper that states the deep learning based recsys is poor. 
3. Search the paper related to RL and see how they formulate the problem as reinforcement learning. Then apply different reinforcement learning techniques.
4. Search papers related to sequential recommender system evaluation; 
5. Learn sequential modeling in Coursera
6. Read all models in the sequential recommender system survey
7. 



**Deep Learning on Recommender System:**

1. Model the nonlinearity of the user-item interaction.
2. Learn useful representation of input data.
3. Make sequential recommendation, i.e. next-item basket prediction and session based recommendation. 





**Ideas:**

- To model the problem as RL, the only agent could be all customers. The updated interaction is the action, and then what is the state and reward ? 
- We may seek helps from the multi-agent reinforcement learning.
- **Every decision or action will reshape the knowledge graph or auxiliary information of agent.** 
- User behavior should not be the action, but the recommendation and user behavior combined be the action in the reinforcement learning. 
- Graph convolutional network can be used for recommendation, then the dynamic graph representation learning should also work for dynamic recommendation.



 

