---
layout: post
title: Activation Functions
categories: article
permalink: /:categories/:title
tags: [DeepLearning, Activation Functions]
progress: False
image: '/assets/images/RNN_example.svg'
---

> Activation functions are used to introduce non linearity in the model.

As we know machine learning is nothing but finding out best possible weights & biases but wait! what is exact role of these weights & biases in machine learning?

To understand **activation functions** we  need to understand role of these weights & biases so, lets first try to understand these weights & biases first. For simplicity we shall stick back to two dimensional.

What is equation of line two dimensional?

$$ y=mx+c $$

![Line](/assets/images/line_eq.svg)


In above equation 'm' is slope and 'c' is intercept, slope is defined as below

$$ slope = \frac{change \; in\:vertical\,direction}{change \; in \; horizontal \; direction} = \frac{\Delta vertical \; direction}{\Delta horizontal \;direction} $$

and intercept is the point where line passes on vertical axis. So simply to say `y=mx+c` the boundary and `m` is direction of the boundary and `c` is location the boundary which separates the data points very well.

Now lets assume our data is not linearly separable that means a simple line cannot separate the data points in that case we need a boundary that is in non linear form like a curve so, here comes the picture of activation function. When a activation function is applied on  `mx+c` we get an output of a non linear structure that may separate the points well.

> Without activation function, model only learns a linear function which may work well on linear data but fails on non-linear data.

## Types Of Activation Functions

###  1.Sigmoid Activation Function

What ever the value you pass to the sigmoid function it transform them between 0 & 1

#### Formula:     

$$ \sigma(z) = \frac {1}{1+exp^(z)} $$

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where  $$  z = w^T . x_i + b $$

![Line](/assets/images/sigmoid_activation_functions.svg)

### 2.Tanh Activation Function
![Line](/assets/images/tanh_activation_functions.svg)

### 4.Relu Activation Function
![Line](/assets/images/relu_activation_functions.svg)

### 5.Leky Rely Activation Function

### 6.Softmax Activation Function



