---
layout: post
title:  "Logistic Regression and Neural Networks - Part 3: Optimization with Forward and Back Propagation!"
date:   2017-08-19 10:25:00 +0530
categories: deeplearning neuralnetworks logisticregression optimization
latexscript: js/katex_render.js
---

Welcome to Part 3 of explaining logistic regression using neural networks! We gave a medium size picture of the whole thing in [Part 1][week-2-part-1] and then defined the optimization problem in [Part 2][week-2-part-2]. In this episode, we'll first develop an algorithm to solve the problem by iterating through the examples, and then use the awesome power of vectorization to go through all examples at once. So, let's get started, yeah?

## Table of Contents
{:.no_toc}
* Do not remove this line (it will not be displayed)
{:toc}

## Recap

Remember the problem? No? Here it is again:

<script type="math/tex; mode=display">
\begin{aligned}
\text{Minimize } J(w, b) &=  -\frac{1}{m} \displaystyle \sum_{i=1}^{m} \left(y_i \log a + (1-y_i) \log(1 - a \right) \\
\text{where: } a(w, b) &= \hat{y} = \sigma(w^Tx + b) = \sigma(z)
\end{aligned}
</script>
Refer to [Part 1][week-2-part-1] (and [Part 2][week-2-part-2] too I guess) if you're unclear on what any of those letters mean.

The algorithm (also developed in [Part 2][week-2-part-2]) was shown to be:

1. Assume starting values for all parameters <script type="math/tex"> (w, b) </script> in our case
2. Calculate the gradient: The gradient is basically <script type="math/tex"> \displaystyle \left(\frac{ \partial J}{\partial w}, \frac{ \partial J}{\partial w} \displaystyle\right) </script>. Note that the derivative <script type="math/tex"> \displaystyle \frac{ \partial J}{\partial w} </script> is a vector with the same size as <script type="math/tex"> w </script> 
3. Update the parameters: 
<script type="math/tex; mode=display">
\begin{aligned}
w &= w - \alpha \frac{ \partial J}{\partial w} \\
b &= b - \alpha \frac{ \partial J}{\partial b}
\end{aligned}
</script>
4. Repeat till convergence, i.e till the values of <script type="math/tex"> w </script> and <script type="math/tex"> b </script> don't change much with new iterations

In the [lectures][deep-learning], Andrew Ng took us through an example where he showed how the gradient was calculated. To simplify things, the feature vector was only of size 2. But we're big girls and boys, so let's have some fun and make our example general, i.e. of size <script type="math/tex"> n_x \times 1 </script>. If that means I have think more as I develop this example, so be it!

## The Big Picture

To give you the big picture, I've made a small picture:

{:refdef: style="text-align: center;"}
![Big picture for optimization](/assets/dl_week2/lr_nn-propagation.png)*Forward propagation and Back Propagation*
{: refdef}

Note the two newly introduced terms, forward propagation and backward propagation. Here's what the terms mean:

1. **Forward propagation**: In this step, the input feature vector is fed through the neuron with the current values of the parameters <script type="math/tex"> (w, b) </script>. The cost function <script type="math/tex"> J </script> is calculated with the fresh values of the outputs. This is where *predictions are made*.
2. **Back Propagation**: Here, the derivatives of <script type="math/tex"> (w, b) </script> with respect to the cost function <script type="math/tex"> J </script> are calculated. Then the values of our parameters are updated.

<mark> A quick note on the calculation of derivatives </mark>: The derivatives are calculated using this *awesome, cool, amazing* technique called a <mark>Computational Graph</mark>. This technique is really cool, and I found someone online who explains it much better than I ever could. Here's [his awesome article][computational-graph]. I highly recommend reading through it. It's much better than the lecture on Computational Graphs in the Coursera course.

## The Calculation of Derivatives

So...I'm not going into too much detail here, because most of this is algebra. It is interesting algebra for sure, but the results are sufficient for this post. 

You can do most of the derivations by hand, or even use Wolfram Alpha. I'm just going to list them out.

Before I do, there is *one* important thing I'd like to mention:

<script type="math/tex; mode=display">
\displaystyle \frac{\partial J}{\partial w_i} = \displaystyle \frac{1}{m} \displaystyle \sum_i^m \displaystyle \frac{\partial \mathbb{L(a^{(i)}, y)}}{\partial w_i}
</script>

The reason this is true is that we **assume** the weights are independent of each other, so *only the <script type="math/tex"> i^{th} </script> weight corresponds to its partial derivative. The reason this is important is that to calculate <script type="math/tex"> \displaystyle \frac{\partial J}{\partial w_i}</script>, we only need to calculate <script type="math/tex"> \displaystyle\frac{\partial \mathbb{L}}{\partial w_i} </script>, which is easily found out.

As promised, here are the significant derivatives involved. All of these are defined for a *single* example:

<script type="math/tex; mode=display">
\begin{aligned}
\displaystyle \frac{\partial \mathbb{J}}{\partial a} &= \frac{1}{m} \displaystyle \frac{\partial \mathbb{L}}{\partial a} = \frac{1}{m} \left( -\frac{y}{a} + \frac{1-y}{1-a} \right) \\
\displaystyle \frac{\partial \mathbb{J}}{\partial z} &= \frac{1}{m} \displaystyle \frac{\partial \mathbb{L}}{\partial z} = \frac{1}{m} (a - y) \\
\displaystyle \frac{\partial \mathbb{J}}{\partial w_i} &= dw_i = \frac{1}{m} \displaystyle \frac{\partial \mathbb{L}}{\partial w_i} = x_i \frac{\partial \mathbb{L}}{\partial z_i}
\end{aligned}
</script>

## The Algorithm for <script type="math/tex"> m </script> examples

I'll lay out algorithm here, to iterate over <script type="math/tex"> m </script> examples. The idea is to repeatedly iterate till the cost function <script type="math/tex"> J </script> converges.

1. Initialize values:
<script type="math/tex; mode=display">
\begin{aligned}
J &= 0 \\
w &= [0, 0, \ldots 0]_{1 \times n_x} \\
b &= 0  
\end{aligned}
</script>

2. <mark> Outer Loop</mark>: For each example <script type="math/tex"> i \in [1,2, \ldots m ]</script>: 
<script type="math/tex; mode=display">
\begin{aligned}
z^{(i)} &= w^T x^{(i)} + b \\
a^{(i)} &= \sigma(z^{(i)}) \\
\mathbb{L}^{(i)} &= - \left( y^{(i)} \log a^{(i)} + (1 - y^{(i)}) \log (1 - a^{(i)}) \right) \\
J = J + \mathbb{L}^{(i)}
dz^{(i)} &= a^{(i)} - y^{(i)} \\
\text{Set } dw &= [0, 0, \ldots 0]_{1 \times n_x}, db = 0 \\
\end{aligned}
</script>
    2.1. <mark>Inner Loop</mark>: For each element <script type="math/tex"> k \in [1, 2, \ldots n_x] </script> 
    <script type="math/tex; mode=display">
    \begin{aligned}
    dw_k &= dw_k + x_k^{(i)} dz^{(i)} \\
    db &= db + dz^{(i)}
    \end{aligned}
    </script>
3. Update (w,b) as:
<script type="math/tex; mode=display">
\begin{aligned}
w &= w + \displaystyle \frac{1}{m} dw \\
b &= b + \displaystyle \frac{1}{m} db \\
\end{aligned}
</script>

3. Repeat steps 1, 2, and 3 till the value of <script type="math/tex"> J </script> converges, i.e. it does not change with more iterations, or changes within a preset small value.

Some explanation of what's going on here is probably required, so here it is:

1. In the first step, we just initialize all the values to zero. Note the dimensions of the parameters. <script type="math/tex"> J \in \mathbb{R}, w \in \mathbb{R}_{1 \times n_x}, b \in \mathbb{R} </script> 
2. We loop through the examples:
    * <mark>Forward propagation: </mark>For each example, we calculate our predictions, and our loss function
    * <mark>Backward propagation: </mark>Then we loop through the individual feature vector for this example to find the contribution of this example to the weights. <mark>This is where information is transferred from the input feature vector to the output parameters</mark>
3. We update our parameters
4. Repeat till convergence.

I hope that's clear. If not, or if you see something wrong here, leave a comment and I'll update the post!

## Vectorization

In computation, and especially while dealing with large amounts of data, it's not very efficient to have these nested for loops in the code. Fortunately, most of the operations in the optimization above are *vectorizable*, i.e. they can be converted to matrix operations.

Why would we want to convert these operations to matrix operations? Well, mathematicians have spent hundreds of years working out cool things related to matrices that make them extremely friendly to fast computation. Computer Scientists have also spent a lot of time on making matrix operations efficient, though not hundreds of years, but only because computers haven't existed that long. 

Python is especially bad at nested loops. Since it's an interpreted language, it can't make any optimizations of its own and becomes super slow. So, we vectorize!

I'm not going into the derivations here again, because again, it's just algebra, and this time, it's not even that complicated. You just need to know a little bit about how matrices work.

But <mark>VERY IMPORTANTLY</mark>, remember that the matrix rules below <mark>are what you are going to use in your code finally!</mark>

### Vectorized Logistic Regression

The matrices and vectors involved are:

<script type="math/tex; mode=display">
\begin{aligned}
X_{n_x \times m} &= [x^{(1)}, x^{(2)}, \ldots, x^{(m)}] \\
y_{1 \times m} &= [y^{(1)}, y^{(2)}, \ldots, y^{(m)}] \\
w_{m \times 1}^T &= [w_1, w_2, \ldots, w_m] \\
Z &= [z^{(1)}, z^{(2)}, \ldots z^{(m)}] \\
  &= w^TX + [b, b, \ldots b]_{1 \times m}
\end{aligned}
</script>

With that, we can calculate:

<script type="math/tex; mode=display">
\begin{aligned}
A &= \sigma(Z) \\
  &= [a^{(1)}, a^{(2)}, \ldots, a^{(m)}] \\
dZ &= A - Y \\
   &= [(a^{(1)} - y^{(1)}), (a^{(2)} - y^{(2)}), \ldots, (a^{(m)} - y^{(m)})] \\
dw &= \displaystyle \frac{1}{m} \displaystyle \sum XdZ^T \\
db &= \displaystyle \frac{1}{m} \displaystyle \sum(dZ)
\end{aligned}

</script>

These matrix operations will replace the inside of the for loop in the algorithm outlined above. We *can* actually replace even the outer for loop, but that will involve some advanced mathematics we will get to later. 

## Summary

So, there we have it. After a long and winding road, we've gone through what it would take to do logistic regression using neural networks. Note that we *still have only one neuron*!

Until next time, adios! Please feel free to leave comments below with questions, or complaints that I've been too vague or too wordy or too crazy.



[deep-learning]: https://www.coursera.org/specializations/deep-learning
[week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}
[tricking-neural-networks]: https://medium.com/@ageitgey/machine-learning-is-fun-part-8-how-to-intentionally-trick-neural-networks-b55da32b7196
[computational-graph]: https://colah.github.io/posts/2015-08-Backprop/
[week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}
[week-2-part-2]: {{ site.baseurl }}{% post_url 2017-08-15-week-2-part-2-lr-gradient-descent-and-neural-networks %}

    
    


