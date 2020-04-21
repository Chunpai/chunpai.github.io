---
layout: post
tags: math
author: Chunpai
---



## TPR, FPR, TNR, and FNR

Denote $X = \{\text{a set of people for testing}\} = S \cup H = P \cup N$ where 

$S = \{ \text{sick people who get lung cancer} \}$,  $H = \{ \text{health people who does not have lung cancer} \}$, $P = \{ \text{people are tested positive for lung cancer} \}$, $N = \{ \text{people are tested negative for lung cancer} \}$. 

$\text{True Positive（真阳）} = S \cap P$, set of people who have lung cancer, and tested as positive

$\text{True Negative （真阴）} = H \cap N$, set of people who does not have lung cancer, and tested as negative

$\text{False Positive（假阳）} = H \cap P$, set of people who does not have lung cancer, but tested as positive

$\text{False Negative （假阴）} = S \cap N$, set of people who have lung cancer, but tested as negative

$TPR = \frac{|S\cap P|}{|S|}$ 

$TNR = \frac{|H\cap N|}{|H|}$ 

$FPR = \frac{|H\cap P|}{|H|}$ 

$FNR = \frac{|S\cap N|}{|S|}$ 

Ideally, we would like the TRP and TNR to be close to 1, and FPR and FNR to be close to 0.



### Accuracy, Precision, Recall, F-Score

$$
\text{Accuracy} = \frac{|S\cap P|+|H\cap N|}{|X|} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data set**, especially when negative size is larger than positive size and we care about TP more. 


$$
\text{Precision} = \frac{|S\cap P|}{|P|} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{|S\cap P|}{|S|} = \frac{TP}{TP + FN} = TPR
$$

$$
\text{F-Score} = \frac{2\times Precision \times Recall}{Precision + Recall}
$$










### Reference

[1] Coursera: Data Science Math Skills

[2]


