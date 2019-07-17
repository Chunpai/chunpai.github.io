---
layout: post
author: Chunpai
---

* TOC
{: toc}




## Recommender System

`2019-07-17`

- read paper *Neural Collaborative Filtering*
- read Hongwei Wang's thesis

## Structured Prediction

`2019-06-24`

- ~~implement the structured SVM, and try to understand the algorithm and limitations~~
  - ~~cutting plane method to find the most violated constraints~~
  - ~~we only need to implement 3 procedures: 1) feature mapping 2) the loss function 3) maximization to find cutting plane~~ 
- ~~use PyStruct module rather than the SVM^{struct} API~~ 
- [ICML 17](http://icml.cc/2017//)[Workshop on Deep Structured Prediction](http://deepstruct.github.io/)
- [NAACL 2019](https://naacl2019.org/)[Workshop on Structured Prediction for NLP](http://structuredprediction.github.io/)
- [Tutorial: AAAI-16: Learning and Inference in Structured Prediction Models](http://cogcomp.org/page/tutorial.201602/)
- [Deep Stuctured Learning Lectures](https://andre-martins.github.io/pages/deep-structured-learning-ist-fall-2018.html)

`2019-06-25`

- extend the PyStruct, and choose one model, learner, inference first. 

  - run examples in PyStruct first
  - We need to know how cutting plane algorithm works, and check which part of PyStruct using cutting plane
  - encode the connectivity constraints into the cutting plane method
  - check the examples of pystruct to do the multiclass svm on MNIST

- Try to understand the implementation of conditional random field, unary potential and pairwise potential

`2019-06-26`

- [a good tutorial on feature functions in CRF](https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)   
- we choose multiclass as model, and n_slack_svm as learner. The learner will call the model, and train. We only need to make modification on prediction step in learner module.
- We need to read the model.fit() function first. 

`2019-06-27`

- how to encode the tree structure with structured svm

- what is unary potential and pairwise potential, how can we use them in the inference ? we can also add higher order potential, global potential.

- do we find the most violated the constraints with fixed parameters ? yes. refers to Yisong Yue's tutorial

- Some materials on understanding training of structured svm:

  -  [A note on structured SVMs.](https://www.dropbox.com/s/tdhawb9n4ouw0q9/2009_Notes_StructuredSVMs.pdf?dl=0) 

  - [PyStruct author's thesis](https://www.dropbox.com/s/icdpico8330yj72/thesis_amueller_conditional_random_fields.pdf?dl=0)

  - [StructSVM Slides from CalTech Structured Prediction Course](https://taehwanptl.github.io/)

  - [Cutting-Plane training of structured svm](https://www.dropbox.com/s/0vgd14mmmaj77r2/Cutting-Plane_Training_of_Structural_SVM_joachims_etal_09a.pdf?dl=0)

  - [Yisong Yue's tutoral on structured svm](https://www.dropbox.com/s/cxkelruwgewxck4/svm_struct_intro_yue_yisong.ppt?dl=0)

- [Generalized IOU vs IOU](<https://giou.stanford.edu/GIoU.pdf>), may need to check the theory behind. 

- we may still use the energy function that encode the structured dependency rather than the pure IoU score, since the IoU or f-measure does not use the structured compatibility score.

- We can train the dvn to output the predicted IoU and energy, but should be regularized by the difference of the energies

- semantic segmentation regularized by bounding box, which is convex hull. We can borrow the idea from the connected subgraph polytope from global connectivity potential for random field models

- how to do inference after training ? replace the pystruct inference step with Graph-IHT.

- can we do multiple inferences in DVN model or output multiple y ?

  

`2019-07-03`

- the structured svm program will output the "inf" value for P matrix when regularizing QP.

  

`2019-07-08`

- how to avoid the 0 cost augmented inference
  - remove the structured projection
  - change the sparsity to enforce to get different output 

## Computer Vision  

- [9 Applications of Deep Learning for Computer Vision](https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/)
- Book: *[Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)*
- 

## Geometry Deep Learing

 `2019-07-03`

- Some tutorials on geometry deep learning:
  - http://www.geometricdeeplearning.com/
  - [Learning Representations via Graph-structured Networks](https://xiaolonw.github.io/graphnn/)

  

## Reinforcement Learning

`2019-06-24:`

- reading of [structured prediction is not reinforcement learning](https://nlpers.blogspot.com/2017/04/structured-prediction-is-not-rl.html) 

`2019-07-12:`

- read paper "Connecting the Dots Between MLE and RL for Sequence Generation"
- deep learning and sequence model https://www.coursera.org/learn/nlp-sequence-models/home/welcome

## Meta Learning



## Time-Series Analysis	

 `2019-07-03`

- [Applied Time Series Analysis](https://newonlinecourses.science.psu.edu/stat510/lesson/1/1.1) 
- [Tutorials on time series analysis](http://dept.stat.lsa.umich.edu/~ionides/tutorials/index.html)
- [Tutorial Point on Time Series Tutorial](https://www.tutorialspoint.com/time_series/)