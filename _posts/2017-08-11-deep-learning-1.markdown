---
layout: post
title:  "Deep Learning Tutorials - Start here!"
date:   2017-08-10 10:25:00 +0530
categories: introduction deeplearning
---

Welcome to my Deep Learning Tutorial Series!

Here are the posts in order so you can find them all from the same landing page:

1. [Introduction to Deep Learning][week-1]: A bird's eye of what deep learning is why it's become so popular these days
2. Logistic Regression and Neural Networks: An exploration of the idea behind Neural Networks (actually, just one neuron for this part) using a Logistic Regression, a a popular Machine Learning Algorithm.
    - [Part 1: The Medium-Size Picture][week-2-part-1]: An introduction to the notation we will be using through this tutorial series, and talk a bit about logistic regression and how it relates to neural networks
    - [Part 2: Defining the Problem][week-2-part-2]: Even more notation, an attempt to explain why we need optimization, a bit about the idea behind gradient descent, and finally the definition of the optimization problem.
    - [Part 3: Optimization with Forward and Back Propagation][week-2-part-3]: A simple example illustrating gradient descent, the idea behind forward and back propagation, the vectorized formulation of the optimization problem.
3. [Shallow Neural Networks][week-3]: An exploration of neural networks with only one, or maybe just a few more, layers. This post will be useful to understand how neural networks work "under the hood" (or rather, as far under the hood as we can see) and just as importantly, how they are represented.

The idea to do this series came after I registered for the new [Deep Learning specialisation][deep-learning] on Coursera. It's taught by Andrew Ng, one of the most well-known names in Machine Learning.

This tutorial series is serving multiple purposes:

1. I understand better the material that is being taught
2. I have written notes for the specialisation
3. Maybe these notes will help someone else
4. I learn to draw Affinity Designer. It's what I'll be using to make the illustrations you see in the coming posts.
5. I learn to use the Jekll blogging platform, because why not?

I intend to use this post as a landing page for the tutorial series. I'll link to the individual posts from here.

[deep-learning]: https://www.coursera.org/specializations/deep-learning
[week-1]: {{ site.baseurl }}{% post_url 2017-08-11-week-1-intro-to-nn %}
[week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}
[week-2-part-2]: {{ site.baseurl }}{% post_url 2017-08-15-week-2-part-2-lr-gradient-descent-and-neural-networks %}
[week-2-part-3]: {{ site.baseurl }}{% post_url 2017-08-19-week-2-part-3-optimise %}
[week-3]: {{ site.baseurl }}{% post_url 2017-08-24-week-3-part-1-shallownnrepresentation %}