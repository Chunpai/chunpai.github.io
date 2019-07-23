---
layout: post
author: Chunpai
---

* TOC
{: toc}
## Reviews of Recommender System

Implicit and Explicit Feedbacks:

Implicit feedback indirectly reflects users' preference through behaviors like watching videos, purchasing products and clicking items. Compared to explicit feedback (i.e. rating and reviews), implicit feedback can be tracked automatically and is thus much easier to collect for content providers. However, implicit feedback is more challenge to utilize and interpret, since user satisfaction is not observed directly. 



Pointwise Loss and Pairwise Loss

The recommendation problem is to estimate the score of unobserved item for users. Pointwise learning is to minimize the squared loss between observed $y_{ui}$ and predicted $\hat{y}_{ui}$. If the target values are binarized, we may use the binary cross entropy loss function. Pairwise learning adopts the idea that observed entries should be ranked higher than the unobserved ones. As such, instead of minimizing the margin between observed entry $y_{ui}$ and $\hat{y}_{ui}$,  pairwise learning ~~also~~ maximizes the margin between observed entry $\hat{y}_{ui}$ and unobserved entry $\hat{y}_{uj}$. Pairwise learning is more difficult to train, and the performance is not stable. However, we may combine the pointwise the pairwise losses. 



Tasks: 

1. NCF implementations in TensorFlow and PyTorch, and check the TensorRec framework.
2.  Check the paper that states the deep learning based recsys is poor. 







 

