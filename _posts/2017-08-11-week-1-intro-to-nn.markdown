---
layout: post
title:  "Introduction to Deep Learning"
date:   2017-08-11 10:25:00 +0530
categories: deeplearning neuralnetworks
latexscript: js/katex_render.js
---

Technically, [Deep Learning][deep-learning-wiki] is the application of Neural Networks where more than one hidden layer of neurons is involved. In the common form that it has pervaded the media today, it also usually involves a mixture of neural networks with other algorithms specifically applied to large datasets in a variety of areas. 

Don't know what that means? You're in the right place! The idea of this post is to cover just enough ground (in a very shallow way, pun sort-of intended) so that you can say the above paragraph out loud and understand what it means. So the following questions that I'm sure are burning your stomach will be answered:

1. What is a neuron?
2. What is a neural network?
3. What is deep learning?
4. Why has it taken off in the last decade or so?

This post corresponds roughly to Week 1 in the [Coursera Deep Learning Specialisation][deep-learning-coursera]. 

So, let's get started!

## What is a Neuron?

In the not-Computer-Science world a neuron is an organic thing in your body that is the basic unit of the nervous system. Any information that travels to your brain -- heartbreak for example -- goes through neurons. The way I understand it (it is important to note here that I am not a biologist), a neuron turns "on" at some level of electrical voltage. A pulse is carried through the neurons to the brain when they are all "on". That's how information -- the feeling you get when you resolve a bug in your code, for example -- gets to your brain.

In the Computer Science world, it's the same, really. A neuron takes an input, applies a mathemtical transformation to it, and then gives an output. 

"Wait, isn't that the same as a mathematical function?", one of you asks.

Nope, nope. A neuron actually does two things to achieve its transformation:

1. It applies a dot product to the inputs with the weights (the weights are a property of the neuron) and adds a threshold. For example, if the input vectors are of type <script type="math/tex"> \vec{x} \in \mathbb{R}_{3 \times 1}  </script>, the weights will be of the form <script type="math/tex"> \vec{w} \in \mathbb{R}_{3 \times 1} </script>, and the output of this step will be <script type="math/tex"> \vec{x} . \vec{w} + b </script>
2. Applies an activation function to the result of 1 above, i.e <script type="math/tex"> f(\vec{x}.\vec{w} + b) </script> where <script type="math/tex"> f </script> is a function chosen by us.

Here is a carefully constructed illutration that shows the in-depth workings of a single neuron as it is commonly represented in Computer Science.

![A not-so-helpful illustration of a neuron]({{ site.url }}/assets/dl_week1/nn_basic.jpg)
*This illustration doesn't illustrate much other than my inexperience with drawing software*

"Okay fine. The function that the neuron applies can be any linear function, right? What is the class of activation function that you can use?", someone pipes in from the back.

That's actually a very good question, thank you!

When neural networks were first being designed and used, the [sigmoid][sigmoid] function was pretty popular. The idea of a sigmoid function is to kinda-sorta simulate the behaviour of a switch (Y = 0 when X is negative and Y = 1 when X is positive), but with the property that is continuous and differentiable everywhere. So if the output of our linear function is positive, you get a value close to 1, and if it's negative, you get a value of 0. Can you see how that will be useful if you are doing a classification problem where you either have to say "Yes" (1) or "No" (0)?

These days however, the RELU (REctified Linear Unit) is much more popular (shown in the ugly figure below). Apparently, it works much better with optimization algorithms such as Gradient Descent 🤷‍, even though there is that obvious discontinuity in the derivative. Maybe we'll figure out how this works some day in the future, you and me.

![REctified Linear Unit]({{ site.url }}/assets/dl_week1/relu.jpg)
*They could have just called it the RLU instead of RELU. Why the silly acronym?*

So, to summarise, neuron takes input X, applies the RELU function to it, and generates Y as the output, which can either zero or positive. All clear? Good!

## Neural Networks

I'm sure many of you saw this coming, but guess what neural networks are? They're networks of neurons!

Neurons can be stacked together in a variety of ways, some of which are mind-bogglingly complex, but thankfully we don't have to think about that yet. Right now, imagine them stacked in layers, each layer feeding into the one ahead of it. Here's a figure for ya if my words aren't that well chosen.

![A feedforward neural network]({{ site.url }}/assets/dl_week1/nn_with_layers.jpg)
*A feedforward neural network. Note that the figure is incomplete. Each neuron can link to ALL neurons in the next layer*

So here's what happens when you have the input vector: 

1. Each neuron in each layer has a set of weights. X is fed into the first layer. Each individual neuron outputs the thresholded dot product, to which RELU is applied.
2. The output of all those neurons are fed to the next layer. 
3. This is repeated till suddenly, you have your output value Y. 

There's some handwaving there, I admit, because this stuff is still simmering in the pot of understanding of my mind. I'll update this post with more clearly chosen words over the next few weeks. 

I know at least some of you are looking at that figure and thinking: How do you know how many layers to use? What do the layers mean? How do we decide the individual functions? Wait, what's happening?

So, here's what *we* decide:

1. The number of layers between the input and output 
2. The number of neurons in each layer
3. The function that each neuron applies. By "decide" here, I mean that each neuron is an RELU. You don't get to choose that for the most part!

<mark>A cool thing</mark>: The algorithm figures out everything else! Everything else, of course, involves the weights on the individual neuron. The middle layers are sometimes called "hidden", because all you care about are X and Y, and the algorithm figures out everything in between

<mark>Another cool thing</mark>: Does that figure look like your brain? I know it doesn't look like mine. It took a genius to make this connection: Walter Pitts. He spent most of his time chasing this idea of modeling the brain based on neural networks and unfortunately is not alive today to see the results of his work. You can read an excellent article about him and his amazing and sad life [here][walter-pitts]

### Applications and Types of Neural Networks

Neural networks are used everywhere these days: from product recommendations to user-click probabilities, from image recognition to self driving cars. There are different types of neural networks, each of which we will get to at different points during this tutorial series:

1. Standard Neural Nets: Like the ones shown in that awesomely drawn figure above
2. Convolutional Neural Nets: Each layer becomes multi-dimensional. Not sure what that means? To be honest, neither am I. We'll figure it out in a future post. For now, know that CNNs are all the rage in image processing these days
3. Recurrent Neural Networks: wWere we make use of sequential patterns in the data, like in natural language. So this is used in speech recognition, language processing, and those kinds of things
4. Custom/Hybrid: Where you have different techniques, you can mix-and-match. Custom NNs are used in complex applications such as self driving cars.

<mark>Important thing</mark>: These days, all the publicity with deep learning is going to cool-sounding things like Image Processing ("Is that a bird? Is it a plane? No, it's Superman!"), speech recognition ("The rain in spain falls mainly in the plain"), and their ilk. The commonality between these problems is that the data that the algorithms use are **unstructured**. The reason they are so popular is that our brain also seems to think in an unstructured form (I know mine does!) and so maybe we can relate better to these problems.

However, great economic value has been obtained by applying NNs to **structured** data as well. One of the areas that has received the greatest bump in awesomeness because of Deep Learning, for example, is Ad Pricing. You know how when you search for an product on Google, suddenly Facebook is showing you ads for the product. Well, someone is choosing to bid to show that ad to you at that moment, and they're basing their decision on many computers running many iterations of Neural Networks on all the data they have on you!

## Deep Learning and Why It's Suddenly So Popular

Deep Learning is just the application of Large Neural Networks to problems with large amounts of data. 

![Performance of Machine Learning Algorithms]({{ site.url }}/assets/dl_week1/data_vs_performance.jpg)
*Performance of Machine Learning Algorithms versus the amount of data available*

If you stare at the figure above long enough, you'll see what's going on. Deep Learning is in top right of that graph. 

It turns out that the performance of standard machine learning algorithms doesn't improve much if you give them more data. It's almost like they've reached the limit of their "intelligence" and giving them more information just doesn't help. So, we turn to neural networks, and keep adding more and more neurons and layers to it and we notice that the performance keeps improving. 

Awesome! Let's just use the biggest neural network with the largest amount of data. 

Do you see where this is going:

1. **Bigger Neural Networks** means more computing power is needed. Such power was not available till the last decade. Now we have supercomputers and all of Google's might, so we have a fighting chance! These days, Google and NVidia and Intel (?) and AMD (?) are all designing chips specifically with Deep Learning in mind.
2. **Large amount of data**: till the advent of the internet, data was siloed and in small amounts. Now with the internet and people writing unnecessary and wordy deep learning tutorials and other people taking needless smartphones photos, there is a deluge of data just waiting to be mined for information. 

Until the last few years, it was just these two factors that was causing the growth in deep learning performance. But now, we have a new entrant to the arena:

3. **Human Creativity!!** Yup, that's right. Deep Learning algorithms won't take your job if you're making the algorithms. It turns out that a simple modification in the algorithms has a *huge* effect on the performance of Neural Networks. For example, one of the most significant bumps in performance was obtained when the Neural Network funciton was switched from the sigmoid to the RELU.

## Conclusion

As I said in the beginning: *Deep Learning is the application of Neural Networks where more than one hidden layer of neurons is involved. In the common form that it has pervaded the media today, it also usually involves a mixture of neural networks with other algorithms specifically applied to large datasets in a variety of areas.*

Now you know what all of that means. *Mic drop*

[deep-learning-wiki]: https://en.wikipedia.org/wiki/Deep_learning
[deep-learning-coursera]: https://www.coursera.org/specializations/deep-learning
[walter-pitts]: http://nautil.us/issue/21/information/the-man-who-tried-to-redeem-the-world-with-logic
[sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function