---
layout: post
title: Decision Trees
categories: article
permalink: /:categories/:title
tags: [Machine Learning, Ensemble Models]
---

# Important Terminology
**1. Root Node:** The starting node of the tree.

**2. Leaf Node:** The final nodes of the tree where path is terminated.

**3. Internal Node:** The node that is neither a root node nor a Internal Node.

```python
import pandas as pd

data = pd.DataFrame({'Outlook':('Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'),
                     'Temperature':('Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Cold'),
                     'Humidity':('High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'),
                     'Wind':('weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'),
                     'Target':('No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No')})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overcast</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overcast</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rain</td>
      <td>Cold</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



# Entropy

$$ E(X) =  -\sum_{i=1}^{K} P(x_i){\log_b}(P(x_i))  $$
                where K is number of classes
