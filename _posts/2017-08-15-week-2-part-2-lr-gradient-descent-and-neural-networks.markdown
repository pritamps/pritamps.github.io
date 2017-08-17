---
layout: post
title:  "[IN PROGRESS] - Logistic Regression and Neural Networks - Part 2: An Introduction to Optimization"
date:   2017-08-15 10:25:00 +0530
categories: deeplearning neuralnetworks logisticregression
latexscript: js/katex_render.js
---

In the [previous post][previous-post], I introduced the basic idea behind logistic regression and the notation for:

1. **One input**: <script type="math/tex"> x \in \mathbb{R}^{n_x} </script>, a feature vector extracted from whatever our data source is, and <script type="math/tex"> n_x </script> is the number of features
2. **One training label**: <script type="math/tex"> y \in {0,1}</script>
3. **The weight and threshold**: <script type="math/tex">(w \in \mathbb{R}^{n_x}, b \in \mathbb{R})</script> are the weight vector and the threshold respectively
4. **The output**: <script type="math/tex"> \hat{y} = \sigma(w^Tx + b) </script> where <script type="math/tex"> \sigma </script> represents the sigmoid function, and <script type="math/tex"> \hat{y} </script> represents the *probability* that <script type="math/tex"> y </script> is 1. For example, in an object recognition problem, <script type="math/tex"> \hat{y} </script> would represent the probability that an object is in an image.

If you need to refresh your memory, or for some reason, you're reading this before [Part 1][week-2-part-1], this would be a great time to click that link and have it open side-by-side with this one!

Now, if you give a kid just one example of a cat, there's no way he'll be able to tell whether the next thing he sees is a cat or not. Or maybe he will? I don't know. Kids are weird. But Machine Learning algorithms are not. They need many examples of cats to be able to tell the difference between a cat and a not-cat. 

So say we *do* have many examples, and of course, based on all our reading, we already know what Logistic Regression is. So how can we use LR to extract information from all these examples, so our final algorithm is like a kid that knows how to recognize cats (but doesn't do much else)?

Let's find out. But first, we need to play the notation game a bit more, because we need to extend the notations to allow for multiple examples. In the [coursera course][deep-learning] that these notes are based on, Andrew Ng uses his own notation, that's a bit different from what I learned in college and what many papers use. I think he's hoping that his notation catches on, but I'm scared it'll fall into the [standards trap][xkcd-standards]. Anyway, since I'm doing his course and you're reading these notes written by me who's doing this course, let's stick to what he says. 

Here we go!

## Notation

1. **Number of examples**: <script type="math/tex"> m \in \mathbb{R} </script> will represent the number of examples, or images we have. Usually we just use <script type="math/tex"> m </script>, but in case we have a need to differentiate between the training set and the test set, we use:
    - The number of training examples is <script type="math/tex"> m_{train} </script> 
    - The number of test data is <script type="math/tex"> m_{test} </script> 
2. **Training set**: The training set for <script type="math/tex"> m </script> training examples is given by:
<script type="math/tex; mode=display">
\displaystyle \left[ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(i)}, y^{(i)}), \ldots (x^{(m)}, y^{(m)}) \right]
</script>
where <script type="math/tex"> (x^{(i)}, y^{(i)})</script> represents the <script type="math/tex"> i^{th} </script> training example and its label. 
3. **Feature matrix**: The feature matrix is just the feature vectors for the individual examples placed, one after another, i.e. 
<script type="math/tex; mode=display">
X = \displaystyle \left[ x^{(1)}, x^{(2)}, \ldots, x^{(i)}, \ldots x^{(m)} \right]
</script>
The shape (or in math language, the order) of <script type="math/tex"> X </script> is <script type="math/tex"> (n_x, m) </script>, i.e <script type="math/tex"> n_x </script> rows and <script type="math/tex"> m </script> columns
3. **Label Matrix**: Similar to the feature matrix, the label matrix is given by:
<script type="math/tex; mode=display">
Y = \displaystyle \left[ y^{(1)}, y^{(2)}, \ldots, y^{(i)}, \ldots y^{(m)} \right]
</script>
The shape of <script type="math/tex"> Y </script> is just <script type="math/tex"> (1, m) </script> because each element is known to be either 0 or 1.

Phew. That's a mouthful isn't it? But you and I, we're going to get used to this notation together. Now, onto the optimisation problem.

## The Big Picture

Let's take stock of what we have: we have <script type="math/tex"> n_x </script> examples, each of which is represented in a feature matrix <script type="math/tex"> X_{n_x \times m} </script>. The <script type="math/tex"> i^{th} </script> column in the matrix corresponds to the feature vector for the <script type="math/tex"> i^{th} </script> example.

Our goal at the end of all this is to predict the label for new feature vector. The way we do this is by training our algorithm to *learn* based on all the information we have, i.e. the training examples. 

Ideally, our algorithm would be perfect and learn everything perfectly. Of course, this is never the case because:

1. All the information we need might not be in the training examples
2. The way we collect information from the training examples might be incorrect or inefficient
3. Have we forgotten this is the real world? Nothing is perfect here!

In fact, it a rare thing to get perfect performance even on the training set!

What does this mean for us? Well, it means we need a way to measure the imperfections, i.e. the quantity of errors we make on our training set while predicting on the training set, i.e. 
<script type="math/tex; mode=display">
J = \displaystyle\sum_{i=1}^{m} \mathbb{L}(y_i- \hat{y_i})
</script>
where <script type="math/tex"> y_i </script> and <script type="math/tex"> \hat{y_i} </script>   are the actual and the predicted label respectively. The function <script type="math/tex"> \mathbb{L} = \mathbb{L}: \mathbb{R} \rightarrow \mathbb{R} </script> is called the **Loss Function**. 

The total error <script type="math/tex"> J </script> is just the sum of the loss function over all the training examples. This total error is called the **Cost Function**. 

### The Loss Function and The Cost Function

How do we choose the loss function. Here, I'll go over this *very* briefly. First of all, from the equation, you should see that the error is positive when there are more error. With that in mind, here are some ideas:

1. Set <script type="math/tex"> \mathbb{} </script> to 0 if the prediction is correct and 1 if it is wrong.
2. Define <script type="math/tex"> \mathbb{L} </script> as the sum of the squared errors: 
<script type="math/tex; mode=display">
\mathbb{L}(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2
</script>
3. <mark>Our Choice for Neural Networks</mark>: Define <script type="math/tex"> f </script> as this weird looking function called the Cross Entropy Loss: 
<script type="math/tex; mode=display">
\mathbb{L}(y, \hat{y}) =  -( y \log\hat{y} + (1-y) \log(1 - \hat{y} )
</script>
The negative sign above is because the part inside the parantheses decreases with increasing <script type="math/tex"> \hat{y} </script>, and we want it to increase. In the lectures in the coursera deep learning course, I recall Andrew Ng saying this is the logistic loss. That is incorrect. The logistic loss is an even more complex function, which we don't use anyway, so I'm omitting it. 

In general, a good loss function is continuous, differentiable, *always positive*, deals with outliers (large deviations and errors), and works well with optimisation algorithms. For our case of Neural Networks, many engineers and scientists before us have chosen the third function above as the best option. So we do, too. It turns out this function works espcially works well with our optimisation algorithm of choice: Stochastic Gradient Descent (woo. Big words!)

So, there we have it: our complicated loss function. Using it, our cost function <script type="math/tex"> J </script> is simply given by: 
<script type="math/tex; mode=display">
J = -\displaystyle\frac{1}{m} \displaystyle\sum_{i=1}^{m} \left(y_i \log\hat{y_i} + (1-y_i) \log(1 - \hat{y_i} \right)
</script>

Now that we have our cost function, our next goal in life is to minimize it. What this means is that we are trying to get the combination of parameters that gives us *the least difference between our predicted values and the ground truth*. 

We will do that by doing some awesome **numerical optimization**. Let's get started!

## Optimization

... COMING UP TOMORROW ...




[xkcd-standards]: https://xkcd.com/927/
[deep-learning]: https://www.coursera.org/specializations/deep-learning
[week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}
[tricking-neural-networks]: https://medium.com/@ageitgey/machine-learning-is-fun-part-8-how-to-intentionally-trick-neural-networks-b55da32b7196
[loss-functions]: https://en.wikipedia.org/wiki/Loss_functions_for_classification