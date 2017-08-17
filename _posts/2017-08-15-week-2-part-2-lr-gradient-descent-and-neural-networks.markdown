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
3. **The weight and threshold**: <script type="math/tex">(w \in \mathbb{R}^{n_x}, b \in \mathbb{R})</script> are teh weight vector and the threshold respectively
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

## Optimisation

Coming soon!


[xkcd-standards]: https://xkcd.com/927/
[deep-learning]: https://www.coursera.org/specializations/deep-learning
[previous-post]: [week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}