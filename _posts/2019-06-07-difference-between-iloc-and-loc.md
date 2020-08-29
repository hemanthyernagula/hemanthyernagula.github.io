---
layout: post
title: Difference Between .iloc and .loc in Pandas
categories: article
permalink: /:categories/:title
tags: [Pandas, DataFrame]
---


>Even though we can access the pandas data frame by the normal way (dataframe\<column_name>.[index], we have another two ways to access the data frame.

1. iloc
2. loc

We shall create a sample data frame by which can understand better about this above-mentioned operations



{% highlight python %}
import pandas as pd

data = pd.DataFrame({"a":[1,2,3,4,5],
                     "b":[6,7,8,9,10]})
data
{% endhighlight %}

Above code creates a data frame with columns a and b as shown in the figure below

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>

<p style="text-align:center; font-style:italic;">**Now let's try to print rows of the data frame using iloc and loc functions**</p>

```python
data.iloc[0]
```




    a    1
    b    6
    Name: 0, dtype: int64




```python
data.loc[0]
```




    a    1
    b    6
    Name: 0, dtype: int64


data.iloc[0] and data.loc[0] both are giving same result? of course yes, in this case, let’s try to **shuffle the data** and then try to print the rows using iloc and loc functions.
To shuffle the data we have the random operation in the numpy library.

```python
import numpy as np

x = np.random.permutation(data.shape[0])
x
```




    array([1, 4, 0, 2, 3])


np.random.permutation(num) gives us the shuffled values up to num, here, in this case, num is the number of rows in the data frame
As we got x as shuffled index values we shall create a new data frame with these index numbers.

<p style="text-align:center; font-style:italic;">**Now new data frame is created with shuffled index values**</p>

```python
data = data.iloc[x]
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
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

Let’s try to print row using loc and iloc functions
>data.loc[1] — This gives the row at element 1 i.e 2 and 4
data.iloc[1] — This gives the row at index element 1 i.e 5 & 10

See below figure for clarification
```python
data.iloc[1]
```




    a    4
    b    9
    Name: 3, dtype: int64




```python
data.loc[1]
```




    a    2
    b    7
    Name: 1, dtype: int64


As we see in image iloc is the operation of locating the index
And loc is the operation of locating the element

**Tip :** 
<p style="float:left;  padding:5px; background-color:red;">:bulb:</p>
<p style="border:solid 1px; padding:10px;">
To remember iloc and loc, even after shuffling the data to access with index number use loc, to access the elements through indexing after shuffling, use iloc.</p>

