---
layout: post
tags: fun
author: Chunpai
---



* TOC
{: toc}
## Information

Name: Chunpai Wang

Email: chunpaiwang@gmail.com



## Datasets

Download the following datasets and corresponding data description files:

- Property Valuation/Assessment Datasets for 2015 & 2018 in NYC
  - [FY15 Database Files by Tax Class - Tax Classes 2, 3 and 4](https://www1.nyc.gov/assets/finance/downloads/tar/tc234_15.zip)
  - [FY18 Database Files by Tax Class - Tax Classes 2, 3 and 4](https://www1.nyc.gov/assets/finance/downloads/tar/tc234_18.zip)
  - [Database Dictionary](https://www1.nyc.gov/assets/finance/downloads/tar/tarfieldcodes.pdf)
- PLUTO Dataset: Extensive land use and geographic data at the tax lot level 
  - [PLUTO Release 18v2.1](https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyc_pluto_18v2_1_csv.zip)
  - [Data Dictionary](https://www1.nyc.gov/assets/planning/download/pdf/data-maps/open-data/pluto_datadictionary.pdf?v=18v21beta)
  - [README file](https://www1.nyc.gov/assets/planning/download/pdf/data-maps/open-data/plutolayout.pdf?v=18v2beta)



## Preprocessing

1. Convert .mdb files to .csv format.

2. Merge 3 datasets above by unique attributes (borough-block-lots)

3. Within the merged dataset, identify properties (unique borough-block-lots)  where the attribute “TXCL” = 2A or 2B for both 2015 and 2018, and output those properties into "*LimitationProperties.csv*". 

   "TXCL" stands for "*AV-TAX CLASS*"

   "2A" = *APARTMENTS WITH 4-6 UNITS*

   "2B" = *APARTMENTS WITH 7-10 UNITS* 

4. Of the “LimitationProperties”, identify the ones where the attribute “CURAVT-ACT” has increased by MORE than 26% from 2015 to 2018. Output properties into “*ReassessedProperties.csv*”.

   “CURAVT-ACT” stands for *Current Actual Assessed Total Value* 




## Data Understanding

Data collection and propcessing is tedious, time-consuming, but very important. We need to have a good understanding of the data size, distribution, features, balance of training and testing datasets, etc.   Good understanding of data with helps from domain expert is very important for model selection. Due to time limit, I only do the trivial processing, which filters out the categorical features and features contains missing values, and normalize the numerical features. 



## Model Selection

We want to reassess the properties which may be over-assessed in 2018 or have potential of rising value in the future. We may use the <u>FY15 Database Files by Tax Class - Tax Classes 2, 3 and 4</u> as well as the <u>PLUTO Release 18v2.1</u> data to fit a model and predict what the 2018 *“AV-CURAVT-ACT”* for the "Reassessed Properties". 



There are many methods, but briefly can be summaried as regression approach and clustering approach: 

- simple ***linear regression*** by using all numerical attributes or *ridge regression* to get a better generalization. 
- however, there are over 300 attributes, we may use ***LASSO*** instead to do prediction based on most relevant attributes. Alternatively, we may apply the dimension reduction on features, such as principle component analysis, etc..
- **decision tree regression** or **random forest regression** can be used to handle the categorical attributes as well as numerical features. 
- **support vector regression** can be used to predict the expected value as well as marginal values, which can be used for anomaly property accessment detection. 
- **k-nearest neighbors**: find the top-k most similar properties in "*LimitationProperties.csv*" based on the attributes in <u>FY15 Database Files by Tax Class - Tax Classes 2, 3 and 4</u> as well as the <u>PLUTO Release 18v2.1</u> , and average the 2018 “AV-CURAVT-ACT” values for these top-k most similar properties. 
- **matrix completion** which is typically used in recommender system, but we can use it to solve our regression problem as well. It has similar effects as applying PCA first and then do the regression, since we use the top-k eigen-features to do the approximation. 
- **deep learning based on features**: we have **X** and y, and we just split the *"LimitationProperties.csv"* into (train, valiataion, test) sets, where test set is just "ReassessedProperties.csv". We can train it simply with gradient descent. 
- If we can build a network of all properties, where every node represents a property, and every edge represents the closeness or adjacency of two properties (If two properties are located very close within a distance threshold, then we connect an edge between them). Then we can apply **graph convolutional network** to do the prediction based on features as well as spatial information. In this way, our prediction is not just based on the properties with AV-TAX CLASS = 2A or 2B (the "LimitationProperties.csv" dataset), but also considering other properties nearby spatially and temporally. 

However, the "ReassessedProperties" may be outliers, thus if we apply regression and clustering on training data which has totally different pattern from the reassessed properties, we will not get good predicted results. 

Due to time limit, I am assuming "ReassessedProperties" have similar hidden patterns as the training data. And I implemented the trivial deep learning method based on numerical features. With only 100 epochs of training, we can get root mean square validation error less than $6,000 on testing data in "ReassessedProperties.csv", however the testing error is extremely high. In some sense, the model is overfitted. Even with fewer training epochs, the deep learning model is still overfitted. One possible reason is I filtered out some important features, and another possible reason is there are only few training properties have similar pattern as the data in "ReassessedProperties". 

I opt to the decision tree regression (still with numerical features). The overall performance is shown in the histogram below. The x-axis is the prediction error, and y-axis presents the number of reassessed properties in corresponding error bin. 



![error](../assets/img/eval-1558314435881.png)



I am sorry for not providing more plots on results and data, but my code is provided. The code is written in python 3.6. I am very busy recently, and only have time to work on this occationally. I am very interested in this real world property appraisal application, and I am excited to continue working on this during this summer. Please let me know if you think I am a good fit. Thanks.








