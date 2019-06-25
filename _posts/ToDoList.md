---
layout: post
author: Chunpai
---

* TOC
{: toc}




## Network Embedding

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
- 

## Reinforcement Learning

`2019-06-24:`

- reading of [structured prediction is not reinforcement learning](https://nlpers.blogspot.com/2017/04/structured-prediction-is-not-rl.html) 

## Meta Learning



​	


