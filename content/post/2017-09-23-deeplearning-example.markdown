
---
title: Deep Learning Example
author: pritamsukumar
type: post
date: 2020-05-29
draft: true
featured_image: 
url: 
categories:
---

Instead of covering the same ground over and over (networks, layers, nodes, you know it all now!), I'm going to jump right into getting the actual code out. 

<mark>Full disclaimer</mark>: This code is based on the assignments from the Coursera deeplearning course. So the structure of the final code I'll develop will be the same as what Andrew Ng and co. use. The structure isn't really the most efficient, but I'm treating this code as only a means to understanding the concepts. We're all going to be using libraries and frameworks, eh?

If any of the following seems new or weird to you, feel free to go back over the previous tutorials and posts. A good place to start would be [the introductory post on this blog][deep-learning-introduction].

$$ \alpha + \beta $$

The main steps to learning with neural networks are:

1. Initialise the nodes and variables
2. Perform Gradient Descent
    1. Forward Propagation
    2. Back Propagation (Is this one word or two?)
3. Pray that everything has gone well
4. Realise that this is the real world and everything has probably *not* gone well, and you'll have to debug
5. Try different things and get good-enough performance

When I say real-world, I really do mean real-world. I'm no longer bound by the confines of the Coursera course, except of course that I'll be copying code I wrote there in the structure they gave me 🤷🏽‍. 

The problem we will be trying to solve is to recognize bird sounds. It's certainly not an easy problem, but I've worked on it briefly already last year. I'll introduce it soon enough, so hang on to your..seats!

[deep-learning-introduction]: {{ site.baseurl }}{% post_url 2017-08-11-deep-learning-1 %}