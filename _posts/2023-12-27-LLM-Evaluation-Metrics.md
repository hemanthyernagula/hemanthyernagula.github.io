---
layout: post
title: Decoding LLM Metrics - A Mathematical Dive into Model Evaluation  🧮📊
categories: article
permalink: /:categories/:title
tags: [GenAI, LLM, Evaluation Metrics]
---

>Finding out best model is always a crucialtask, because just because of using wrong/improper evaluation metric we may sometimes end up with missing out best model that is performing out there!


<!-- # Why do we need evaluation metric
- To choose best model.
- Improve the performance of model.
- Keep track of the model when there is variance at inference time. -->


## Why do we need evaluation metric?

An evaluation metric is a way to measure the quality or performance of a language model. It helps us to compare different models and choose the best one for our task. It also helps us to improve the model by identifying its strengths and weaknesses. Finally, it helps us to monitor the model when it is deployed and detect any changes or errors in its output.

There are many types of performance metrics, few of them are as mentioned below
- Manual Evaluation 🧠✨
- Bleu (📈⬆️ Higher is better)
- Rouge (📈⬆️ Higher is better)
- Perplexity (📉⬇️ Lower is better)

<!-- 
{% highlight python %}
from torch import nn
import torch

{% endhighlight %} -->


## Manual Evaluation (Human Evaluation):
- Manual evaluation can provide valuable and subjective feedback on the LLM’s performance. 
- Hiring human evaluators who are domain experts and providing clear guidelines to get best model is one way to evaluate the models.
- But it can also be time-consuming, expensive, and prone to bias.

## BLEU Score

The BLEU (Bilingual Evaluation Understudy) score is a metric for evaluating a generated sentence to a reference sentence. It works by counting matching n-grams in the generated translation to n-grams in the ground truth translation.
- BLUE score is directly proportional to the model performance(i.e higher the score better the model is).
- The score ranges from 0 to 1.
- Generally used for text translations.


Mathematically, the BLEU score is given as follows:

$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

Where:
- $$BP$$ stands for Brevity Penalty, which penalizes the score when the Machine Translation is too short compared to the Reference (ground truth) translations.
- $$N$$ is the maximum order of n-grams used in the calculation. By default, the BLEU score uses up to 4-grams, so $$N$$ would be 4.
- $$w_n$$ are the weights for each n-gram order. By default, the weights are all equal to $$1/N$$, so for BLEU-4, each weight would be 0.25.
- $$p_n$$ is the precision for n-grams of order $$n$$. It is calculated as the number of matching n-grams in the generated and reference translations, divided by the total number of n-grams in the generated translation.

The Brevity Penalty is calculated as follows:

$$
BP =
\begin{cases}
1 & \text{if } c > r \\
\exp\left(1 - \frac{r}{c}\right) & \text{if } c \leq r
\end{cases}
$$

Where:
- $$c$$ is the length of the generated translation.
- $$r$$ is the effective ground truth length. This is usually the length of the ground truth translation that is closest in length to the candidate translation.

### Explanation:

   let's consider an example to understand the best case and worst case scenarios for bigram precision ($$p_n$$) in the context of BLEU score calculation.

  Let's say we have the following candidate translation and reference translation:

  - Candidate Translation (C): "The cat sat on the mat"
  - Reference Translation (R): "The cat is sitting on the mat"

  We first need to generate the bigrams for both the candidate and reference translations:

  - Bigrams in C: {"The cat", "cat sat", "sat on", "on the mat"}
  - Bigrams in R: {"The cat", "cat is", "is sitting", "sitting on", "on the mat"}

  Now, let's calculate the bigram precision ($$p_n$$):

  - Count of bigrams in C that are present in R: 2 ("The cat", "on the mat")
  - Total count of bigrams in C: 4

  So, $$p_n$$ = 2/4 = 0.5

<table>
  <thead>
    <tr>
      <th>Reference Text</th>
      <th>Model Prediction</th>
      <th>pn</th>
      <th>wn</th>
      <th>exp()</th>
      <th>BLEU Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>The cat is on the mat.</td>
      <td>The cat is on the mat.</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>The cat is on the mat.</td>
      <td>There is a cat on the mat.</td>
      <td>0.83</td>
      <td>0.86</td>
      <td>0.84</td>
      <td>0.78</td>
    </tr>
    <tr>
      <td>The cat is on the mat.</td>
      <td>A cat is sitting on a mat.</td>
      <td>0.5</td>
      <td>0.86</td>
      <td>0.65</td>
      <td>0.47</td>
    </tr>
    <tr>
      <td>The cat is on the mat.</td>
      <td>On the mat is a cat.</td>
      <td>0.25</td>
      <td>0.86</td>
      <td>0.46</td>
      <td>0.17</td>
    </tr>
    <tr>
      <td>The cat is on the mat.</td>
      <td>The dog is under the table.</td>
      <td>0</td>
      <td>0.86</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>





<table>
  <thead>
    <tr>
      <th>Reference Text</th>
      <th>Model Prediction</th>
      <th>pn</th>
      <th>wn</th>
      <th>exp()</th>
      <th>BLEU Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>The cat is on the mat.</td>
      <td>News paper in front of my door.</td>
      <td>0</td>
      <td>0.86</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>


## Rouge Score
ROUGE score measures the overlap of n-grams, words, or sentences between the machine output and the reference. There are different types of ROUGE scores, such as ROUGE-N, ROUGE-L, and ROUGE-S, that use different levels of granularity to compare the texts. Here are the formulas for each type of ROUGE score:

1. ROUGE-N: This type of ROUGE score measures the overlap of n-grams (sequences of n words) between the machine output and the reference. The formula for ROUGE-N is as follows¹²:

    $$
    \text{ROUGE-N} = \frac{\sum_{s \in \text{References}} \sum_{\text{n-gram} \in s} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{s \in \text{References}} \sum_{\text{n-gram} \in s} \text{Count}(\text{n-gram})}
    $$

    Where:

    - $$s$$ is a reference summary.
    - $$\text{n-gram}$$ is a sequence of n words in $$s$$.
    - $$\text{Count}_{\text{match}}(\text{n-gram})$$ is the maximum number of times the n-gram appears in both the machine output and the reference summary.
    - $$\text{Count}(\text{n-gram})$$ is the number of times the n-gram appears in the reference summary.


2. ROUGE-L: This type of ROUGE score measures the longest common subsequence (LCS) of words between the machine output and the reference. The LCS is the longest sequence of words that appears in the same order in both texts. The formula for ROUGE-L is as follows¹²:

    $$
    \text{ROUGE-L} = \frac{\sum_{s \in \text{References}} \text{LCS}(\text{Machine Output}, s)}{\sum_{s \in \text{References}} \text{Length}(s)}
    $$

    Where:

    - $$s$$ is a reference summary.
    - $$\text{LCS}(\text{Machine Output}, s)$$ is the length of the longest common subsequence of words between the machine output and the reference summary.
    - $$\text{Length}(s)$$ is the number of words in the reference summary.


3. ROUGE-S: This type of ROUGE score measures the overlap of skip-bigrams between the machine output and the reference. A skip-bigram is a pair of words that can be separated by any number of words in between. The formula for ROUGE-S is as follows¹²:

    $$
    \text{ROUGE-S} = \frac{\sum_{s \in \text{References}} \sum_{\text{skip-bigram} \in s} \text{Count}_{\text{match}}(\text{skip-bigram})}{\sum_{s \in \text{References}} \sum_{\text{skip-bigram} \in s} \text{Count}(\text{skip-bigram})}
    $$

    Where:

    - $$s$$ is a reference summary.
    - $$ \text{skip-bigram} $$ is a pair of words in $$s$$ that can be separated by any number of words in between. 
    - $$ \text{Count}_{\text{match}}(\text{skip-bigram}) $$ is the maximum number of times the skip-bigram appears in both the machine output and the reference summary.
    - $$ \text{Count}(\text{skip-bigram}) $$ is the number of times the skip-bigram appears in the reference summary.



## Perplexity
Perplexity is a measure of uncertainty in the value of a sample from a discrete probability distribution. Perplexity is often used to evaluate the performance of language models, such as those used for machine translation or text generation. A lower perplexity indicates a better fit of the model to the data².

The formula for perplexity of a discrete probability distribution $$p$$ is as follows:

$$
\text{Perplexity}(p) = 2^{H(p)}
$$

Where:
- $$H(p)$$ is the entropy (in bits) of the distribution, and $$x$$ ranges over the events. The entropy of a distribution measures the average amount of information or surprise contained in each outcome. The formula for entropy is:

$$
H(p) = -\sum_{x} p(x) \log_2 p(x)
$$

The base of the logarithm need not be 2, but the perplexity is independent of the base, provided that the entropy and the exponentiation use the same base.

In the context of natural language processing, perplexity is used to measure how well a language model predicts a sample of text. A language model is a probability distribution over sequences of words or tokens. The perplexity of a language model $$q$$ on a test sample $$x_1, x_2, ..., x_N$$ is defined as:

$$
\text{Perplexity}(q) = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 q(x_i | x_1, ..., x_{i-1})}
$$

Where:
- $$ 
    q(x_i | x_1, ..., x_{i-1})
$$ is the conditional probability of the next word $$x_i$$ given the previous words $$x_1, ..., x_{i-1}$$ according to the language model $$q$$. This formula can be derived from the definition of perplexity and the chain rule of probability.

The perplexity of a language model can be interpreted as the weighted average branching factor of the language. The branching factor is the number of possible next words that can follow a given word or sequence of words. A lower perplexity means that the language model is more confident and accurate in predicting the next word, and thus the language has a lower branching factor. A higher perplexity means that the language model is more uncertain and inaccurate in predicting the next word, and thus the language has a higher branching factor.


### Metric Comparison Table

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Purpose</th>
      <th>Score Interpretation</th>
      <th>Mathematical Formula</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Manual Evaluation</td>
      <td>Subjective human feedback</td>
      <td>Qualitative assessment</td>
      <td>-</td>
    </tr>
    <tr>
      <td>BLEU Score</td>
      <td>Text translation evaluation</td>
      <td>Proportional to model performance</td>
      <td>$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$</td>
    </tr>
    <tr>
      <td>Rouge Score</td>
      <td>N-gram, word, sentence overlap</td>
      <td>>Proportional to model performance</td>
      <td>

            $$
            \text{ROUGE-S} = \frac{\sum_{s \in \text{References}} \sum_{\text{skip-bigram} \in s} \text{Count}_{\text{match}}(\text{skip-bigram})}{\sum_{s \in \text{References}} \sum_{\text{skip-bigram} \in s} \text{Count}(\text{skip-bigram})}
            $$

      </td>
    </tr>
    <tr>
      <td>Perplexity</td>
      <td>Language model performance</td>
      <td>Lower is better</td>
      <td>$$\text{Perplexity}(q) = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 q(x_i | x_1, ..., x_{i-1})}$$</td>
    </tr>
  </tbody>
</table>


### Referances
##### BLEU
(1) A Gentle Introduction to Calculating the BLEU Score for Text in Python. https://machinelearningmastery.com/calculate-bleu-score-for-text-python/.
(2) NLP - BLEU Score for Evaluating Neural Machine Translation - GeeksforGeeks. https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/.
(3) Understanding Bleu Score - OpenGenus IQ. https://iq.opengenus.org/bleu-score/.
(4) Bleu Score in NLP - Scaler Topics. https://www.scaler.com/topics/nlp/bleu-score-in-nlp/.
##### Rouge
(1) An intro to ROUGE, and how to use it to evaluate summaries. https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/.
(2) Text Summarization:. How To Calculate Rouge Score - Medium. https://medium.com/mlearning-ai/text-summarization-84ada711c49c.
(3) Understanding BLEU and ROUGE score for NLP evaluation. https://medium.com/@sthanikamsanthosh1994/understanding-bleu-and-rouge-score-for-nlp-evaluation-1ab334ecadcb.
(4) Mastering ROUGE Matrix: Your Guide to Large Language Model Evaluation .... https://dev.to/aws-builders/mastering-rouge-matrix-your-guide-to-large-language-model-evaluation-for-summarization-with-examples-jjg.
(5) Evaluate translation or summarization with ROUGE similarity score .... https://www.mathworks.com/help/textanalytics/ref/rougeevaluationscore.html.

##### Perplexity
(1) Perplexity - Wikipedia. https://en.wikipedia.org/wiki/Perplexity.
(2) Two minutes NLP — Perplexity explained with simple probabilities. https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584.
(3) Perplexity: a more intuitive measure of uncertainty than entropy. https://mbernste.github.io/posts/perplexity/.
