---
layout: post
title:  "Logistic Regression and Neural Networks - Part 1: Setting Up the Problem"
date:   2017-08-11 10:25:00 +0530
categories: deeplearning neuralnetworks logisticregression
latexscript: js/katex_render.js
---

In this post, we will go over the basics of the functioning of a neural network. The idea will be to use Logistic Regression and Gradient Descent to illustrate the fundamentally important concepts of **forward propagation** and **backpropagation**. As an example, we might write some code for image recognition, which should give you an idea of just how powerful neural networks.

The post loosely follows (with some edits and additions by me) the lectures in Week 2 of the [Deep Learning Specialisation on coursera][deep-learning-coursera]. The lectures in Week 2 covered a lot of ground (mostly because they glossed over a lot of cook stuff), so I'm splitting this tutorial into two parts. 

In the first part, I'll introduce the notation (always a pain) and Logistic Regression itself.

Throughout this tutorial, we'll be using the <mark>cat-or-not problem</mark> to illustrate the mathematical and algorithmic points made. The problem: given an image, the network should be trained to be able to say if there is a cat in it or not; i.e. a simple binary classification problem. 

## A brief introduction to Logistic Regression

Logistic Regression is an algorithm that was developed for binary classification. Let's get with our cat problem to get comfortable with the ideas behind the algorithm, the notations used, and all that jazz. The parameters involved in Logistic Regression are:

* What it takes in: 
    - **Feature vectors**: The feature vector is represented as <script type="math/tex"> x \in \mathbb{R^{n_x}} </script>, where <script type="math/tex"> n_x </script> is the number of features. In code this would become an array of dimensions <script type="math/tex"> (n_x, 1) </script>
    - **Training labels** The training label is represtend by <script type="math/tex"> y \in {0, 1} </script>. For example, in our cat-or-not game, <script type="math/tex"> y = 1 </script> would mean that a cat is in the image and <script type="math/tex"> y = 0 </script> would indicate that it is not
* What it calculates:
    - **The weights and the threshold**: <script type="math/tex"> w \in \mathbb{R^{n_x}} </script> and <script type="math/tex"> b \in \mathbb{R} </script>. So <script type="math/tex"> w </script> is an array of dimensions <script type="math/tex"> (n_x, 1) </script> (same as <script type="math/tex"> x </script>), while <script type="math/tex"> b </script> is just a real number
- What it predicts: <script type="math/tex"> \hat{y} = P( y = 1 | x)</script>, i.e. the probability that <script type="math/tex"> y </script> is 1 given <script type="math/tex"> x </script>. 

### Generating features and labels for the Cat-Or-Not problem

In the Cat-Or-Not problem, what we are given for training is a set of images, for each of which has been labelled as having a cat or not. We need to convert our image and our knowledge of whether it has a cat into actual values of <script type="math/tex"> x </script> and <script type="math/tex"> y </script>. This is how we accomplish this in code:

1. We read in the image using one of python's packages (I recommend `ndimage`), and we get an array with size <script type="math/tex"> (r_x, r_y, 3) </script> where <script type="math/tex"> (r_x, r_y) </script> is the resolution of the image (the number of pixels along the two axes) and the 3 values for each pixel represent the RGB color values that the image needs to decide the colour at that pixel. Since <script type="math/tex"> x </script> is a 1-D vector, we convert this 3-D matrix into a 1-d vector, by simply concatenating all of the values into one long vector of dimension <script type="math/tex"> r_x \times r_y \times 3 </script>. 
2. The label <script type="math/tex"> y = 0 </script> if there isn't a cat and <script type="math/tex"> y = 1 </script> if there is a cat. This part is not that complicated.

So now, for each of our images, we have a vector of dimenstion <script type="math/tex"> (r_x \times r_y \times 3, 1) </script>, and a value for <script type="math/tex"> y </script>. 

### Logistic Regression as a Neuron

The problem statement of LR is:

<script type="math/tex; mode=display">
\text{Given }x, \text{ get } \hat{y} = P( y = 1 | x )
</script>

In plain words for our cat-or-not game: given an image represented by the feature vector <script type="math/tex"> x </script>, tell me the probability that there is a cat in it.

We have multiple (many!) images for which we know the "ground truth", i.e. whether the image contains a cat. So we want to train our algorithm so that we best understand from these images what it means for an image to have a cat. Is that sort of clear? The goal of logistic regression is to **minimize the error** between its predictions and the ground truth in the training data. 

We start by defining the prediction <script type="math/tex"> \hat{y} </script> as follows:

<script type="math/tex; mode=display">
\hat{y} = \sigma (w^Tx +b)
\text{    where } \sigma(z) = \displaystyle \frac{1}{1 + e^{-z}}
</script>

"What the hell is that? Where did the <script type="math/tex"> \sigma</script> come from? What is it?", one of you asks.

The idea behind the **sigmoid** function is as follows: <script type="math/tex"> w^Tx + b </script> is a linear function of <script type="math/tex"> x </script>, so that's cool. But this linear function is unbounded, and since we want a probability we have to constrain it to the interval <script type="math/tex"> [0, +1] </script>. As you can see in the image below, the sigmoid is bounded between 0 and 1.

![The Sigmoid Function]({{ site.url }}/assets/dl_week2/sigmoid.png)
*The sigmoid function. Notice how it is 0 for large negative values of <script type="math/tex"> x </script>, 1 for large positive values, and 0.5 when <script type="math/tex"> x = 0 </script>*

So we have a nice measure for probability. It also helps that the sigmoid function is continuous and smooth everywhere, but that's too much for this article.

Can you see the similarity to Neural Networks now? We have a linear transformation and an activation function being applied to an input: exactly like a neuron! Want a figure? Here you go!

{:refdef: style="text-align: center;"}
![Logistic Regression on One Training Example as a Neuron]({{ site.url }}/assets/dl_week2/lr_nn.jpg)
*Logistic Regression on a single training example as a Neuron*
{: refdef}

## Conclusion of Part 1

The following facts are important to keep in mind:

1. Everything so far has been for a *single* training example
2. Our goal is to find <script type="math/tex"> \hat{y} </script>. To find <script type="math/tex"> \hat{y} </script>, we have to calculate <script type="math/tex"> w </script> and <script type="math/tex"> b </script>. 

We don't seem any closer to this than we started, I know. But this post was just to set up the problem and notation.

<mark>Very important thing</mark>: We are trying to minimize the error between our predictions and the ground truth. Put another way, we are trying to *extract as much relevant information* as possible from the training examples, so that the predictions that our Logistic Regression algorithm makes are sensible. I've always found it very useful to think in terms of extracting information from the training examples, and this is a point I'll keep returning to as we go on with this tutorial series

## To be continued...

In the next part, we'll explore: 

1. How we connect all the training examples through an iterative process in order to EXTRACT ALL THE INFORMATION! As you can see, I'm pretty excited about it
2. How we define the deviation between our predictions and the ground truth (the error)
3. How we minimize it

It turns out none of this is easy, but all of it is supremely fascinating.

[deep-learning-coursera]: https://www.coursera.org/specializations/deep-learning
