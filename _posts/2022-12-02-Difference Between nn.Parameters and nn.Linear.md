---
layout: post
title: Difference Between nn.Parameters and nn.Linear
categories: article
permalink: /:categories/:title
tags: [PyTorch]
---
Difference Between nn.Parameters and nn.Linear


>Stright forward answer for this question is both the methods provides same output but... 
The Parameters will add trainable weights to the given vector, i.e nn.Parameter is wrapper to a vector 

# Lets see an example`


{% highlight python %}
from torch import nn
import torch

{% endhighlight %}

Lets say there is an vector of shape 5X2,  If we want to make this w_ vector to be trainable i.e we should be able to apply back propagations, then we have wrap this vector with nn.Parameters as shown below

{% highlight python %}

w_ = torch.FloatTensor(5, 2)
print(w_)
print(w_.shape)


W = nn.Parameter(w_)
W
{% endhighlight %}

```
output:-

tensor([[ 2.7591e-02,  3.0774e-41],
        [-1.9336e+02,  4.5750e-41],
        [ 4.4842e-44,  0.0000e+00],
        [ 1.1210e-43,  0.0000e+00],
        [ 3.6767e+20,  3.0778e-41]])
torch.Size([5, 2])


Parameter containing:
tensor([[3.9058e-02, 3.0774e-41],
        [2.7396e-02, 3.0774e-41],
        [4.4842e-44, 0.0000e+00],
        [1.1210e-43, 0.0000e+00],
        [3.0101e+20, 3.0778e-41]], requires_grad=True)
```


`If you observe the W metrix has parameter called requires_grad, but previous w_ does not have`

Now to train this W we just need to multiply with required vector as shown below

{% highlight python %}

vec = torch.randn(10, 5)
print("vector",vec)


manual_wt = vec.unsqueeze(0).bmm(W.unsqueeze(0)).squeeze()
print("manual_wt", manual_wt.shape)

{% endhighlight %}

```
output:-
vector 
tensor([[-1.4777, -1.7305, -0.4353, -1.0892,  0.2154],
        [-0.4027, -0.5443,  0.3094,  1.0402, -0.9318],
        [ 0.8985, -0.8716, -0.7928, -0.5932,  0.6017],
        [ 0.2847, -2.5082, -1.2951,  0.1136, -1.6509],
        [-0.9043,  0.5854,  1.5851, -0.3034,  0.2532],
        [-0.2623,  0.8554, -1.5242,  0.0248,  0.3215],
        [ 0.1314,  1.1035,  0.3485,  0.9364, -0.3890],
        [ 0.3815,  0.1583, -0.1855,  0.5716, -0.4829],
        [-1.2311,  1.0750, -1.9082,  0.3070,  1.8087],
        [-1.4859, -0.1095, -0.2550, -0.7027, -0.6401]])

manual_wt 
torch.Size([10, 2])


```
Lets see how this operation is equivalent to nn.Linear

{% highlight python %}

lin_vec = nn.Linear(5, 2, bias=False)(vec)

print(lin_vec.shape)


print("shape of both are same",manual_wt.shape == lin_vec.shape)

{% endhighlight %}

```
Output:-
tensor([[-0.5080, -1.3123],
        [-0.7342, -0.1703],
        [ 0.1832, -0.1991],
        [-1.3580, -1.2401],
        [ 0.4050,  0.0766],
        [-0.0567,  0.1869],
        [ 0.0048,  0.6505],
        [-0.2096,  0.2060],
        [ 0.0861,  0.3509],
        [-0.4425, -0.7659]], grad_fn=<MmBackward0>)

shape of both are same True
```

## Hence Proved ;)
