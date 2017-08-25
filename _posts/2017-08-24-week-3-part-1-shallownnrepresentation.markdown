---
layout: post
title:  "Shallow Neural Networks Part 1: Non-code"
date:   2017-08-24 10:25:00 +0530
categories: deeplearning neuralnetworks
latexscript: js/katex_render.js
---

Before we get neck-deep into deep neural networks, let's wade into shallow waters and use a two layer network (one hidden layer, one output layer) to explore properites of neural networks in general. Let's see how the notation extends to multiple layers, and what it means for our matrix computations and math.

Here's the mother-diagram for the rest of this post.

{:refdef: style="text-align: center;"}
![Shallow Neural Network Representation]({{ site.url }}/assets/dl_week3/shallow_nn.png)
*Why the ugly diagram? I just bought one of those drawing tables, so I'm learning how to use it!*
{: refdef}

I'm going to do something silly, just because I can. Even though you probably understand this diagram already, I'm going to build it from the ground up. For even more understanding. Not sure what I mean? Well, read on!

## Table of Contents
{:.no_toc}
* Do not remove this line (it will not be displayed)
{:toc}

## The Layers

{:refdef: style="text-align: center;"}
![Layers in the Shallow Neural Network]({{ site.url }}/assets/dl_week3/layers.png)
*Just the layers*
{: refdef}

As you can see, this neural network has three layers:

1. The input layer (<script type="math/tex"> l = 0 </script>): contains the input vector <script type="math/tex"> x </script> 
2. The hidden layer (<script type="math/tex"> l = 1 </script>): contains the neurons that do the neural network magic
3. The output layer (<script type="math/tex"> l = 2 </script>): gets us our output <script type="math/tex"> \hat{y} </script> 

But in common terminology, we ignore the input layer when counting (that's why <script type="math/tex"> l = 0 </script> for the input layer), but we do count the output layer. So this network is a <mark>two layer neural network</mark>.

Notation: <script type="math/tex"> L </script> is the total number of layers and <script type="math/tex"> l </script> can refer to any individual layer, i.e. <script type="math/tex"> l \in {0, 1, \ldots L} </script>. In our example neural network above, <script type="math/tex"> L = 2 </script>.

### Why is the hidden layer called a hidden layer?

The reason that the hidden layer is called "hidden" is that <mark>we don't see the values the weights there get during training</mark>. After the network is trained, we input <script type="math/tex"> x </script>, and get our predicted label <script type="math/tex"> \hat{y} </script>. We don't know what all the layers in between are doing. As far as we're concerned, they're *hidden*.

## The Activations

{:refdef: style="text-align: center;"}
![Activations in the Shallow Neural Network]({{ site.url }}/assets/dl_week3/activations.png)
*Layers and activations*
{: refdef}

Each neuron behaves like we've already examined [before][week-2-part-1]: it applies a linear transformation ( <script type="math/tex"> z = w^Tx + b </script> ) and then an activation function to it ( <script type="math/tex"> a(z) = g(w^Tx + b) </script>). With that in mind, we are ready to introduce our new notation for a node. 

For <mark>an individual node <script type="math/tex"> i </script> in layer <script type="math/tex"> l </script></mark>, the activation function is given by:

<script type="math/tex; mode=display">
\begin{aligned}
a^{[l]}_i &= g (z^{[l]}_i) \\
          &= g (w^{[l]^T}_i x + b^{[L]}_i)  
\end{aligned}
</script>

## Connecting the Layers
With individual nodes done, we turn to connecting the layers. The notation here now develops so that <mark>for layer  <script type="math/tex"> l </script></mark>, <script type="math/tex"> a^{[l]} </script> represents the vector of all the individual activations,  <script type="math/tex"> w^{[l]} </script> represents that *matrix* of all weights and so on. Thus, we can write: 

<script type="math/tex; mode=display">
a^{[l]} = g (W^{[l]} a^{[l-1]}+ b^{[l]})
</script>

Note how I slyly slipped in <script type="math/tex"> a^{[l-1]} </script> instead of <script type="math/tex"> x </script> in there. If you think about it, it makes sense, because the <script type="math/tex"> x </script> for each layer is just the output of the layer that came before it. afsdf

The weight vector got upgraded to a capital <script type="math/tex"> W </script> because it's now a matrix. Note that the <mark>Weight Matrix has the weight vectors stacked row-wise instead of column-wise</mark>. This is beacuse, as far as I can tell, because Andrew said so in the course, and it makes for easier multiplication, i.e.

<script type="math/tex; mode=display">
W^{[l]}_{n_l \times n_{l-1}} = \begin{bmatrix}
w^{[1]^T}_1 \\
w^{[1]^T}_2 \\
. \\
.\\
w^{[1]^T}_{n_l}
\end{bmatrix}
</script>

The dimensions <script type="math/tex"> n_l </script>  and <script type="math/tex"> n_{l-1} </script> refer to the number of nodes/neurons in layers <script type="math/tex"> l </script> and <script type="math/tex"> l-1 </script> respectively. This will become clearer as we move through the rest of this post.

Let's see what all this means for our specific neural network.

### Layer 0 to Layer 1

{:refdef: style="text-align: center;"}
![Activations in the Shallow Neural Network]({{ site.url }}/assets/dl_week3/layer0to1.png)
*Layers and activations*
{: refdef}

The arrows are connected. Our input is of size <script type="math/tex"> x \in \mathbb{R}_{3 \ times 1} </script>, i.e <script type="math/tex"> n_x = 3 </script>. Layer 1, our hidden layer is of size <script type="math/tex"> n_h^{[1]} = 4 </script>. Note the notation introduced here: <script type="math/tex"> n_h^{[l]} </script> is the number of nodes in hidden layer <script type="math/tex"> l </script>. So, we can write out, with dimensions:

<script type="math/tex; mode=display">
\begin{aligned}
z^{[1]}_{4 \times 1} &= W^{[1]}_{ 4 \times 3} a^{[0]}_{3 \times 1} + b^{[1]}_{4 \times 1} \\
\text{where } a^{[0]} &= x
\end{aligned}
</script>

Using the activation functions, we get;

<script type="math/tex; mode=display">
a^{[1]}_{4 \times 1} = g^{[1]} (z^{[1]})
</script>

where <script type="math/tex"> g^{[1]} </script> represents the array of activation functions for the first layer. Yes, this means each individual node gets its own activation function, a fact I'm conveniently glossing over for the purposes of this post. I'll talk about it in another post.

Now, let's move on to the next layer!

### Layer 1 to Layer 2

{:refdef: style="text-align: center;"}
![Activations in the Shallow Neural Network]({{ site.url }}/assets/dl_week3/layer1to2.png)
*Layers and activations*
{: refdef}

The activations and output of the second layer become:

<script type="math/tex; mode=display">
\begin{aligned}
z^{[2]}_{1 \times 1} &= W^{[2]}_{ 1 \times 4} a^{[1]}_{4 \times 1} + b^{[2]}_{1 \times 1} \\
\hat{y} &= a^{[2]} = g^{[2]} (z^{[2]})
\end{aligned}
</script>

## The Full Network
That's it! That's our full network. So, to summarize, the equations are:

<script type="math/tex; mode=display">
\begin{aligned}
z^{[1]}_{4 \times 1} &= W^{[1]}_{ 4 \times 3} a^{[0]}_{3 \times 1} + b^{[1]}_{4 \times 1}  \text{ where } a^{[0]} = x \\
a^{[1]}_{4 \times 1} &= g^{[1]}(z^{[1]}) \\
z^{[2]}_{1 \times 1} &= W^{[2]}_{ 1 \times 4} a^{[1]}_{4 \times 1} + b^{[2]}_{1 \times 1} \\
\hat{y}_{1 \times 1} &= a^{[2]}_{1 \times 1} = g^{[2]} (z^{[2]})
\end{aligned}
</script>

You might be wondering why I'm insistently putting the sizes on there. Well, it's because these matrix sizes are my Achilles Heel. I get confused with every aspect of them: rows and columns, sizes, dot products, multiplications. So I have to be careful. If you see something wrong there, let me know!

## Training With Multiple Examples

If you hadn't noticed so far, let me be the one to remind you that everything we did so far was for one training example. But of course, for our neural network, we have <script type="math/tex"> m </script> training example, i.e. it's matrix time! We've done most of this in 

Our training matrix <script type="math/tex"> X </script> is just the individual feature vectors stacked next to each other:

<script type="math/tex; mode=display">
\begin{aligned}
X_{3 \times m} &= \displaystyle \left[ x^{(1)}_{3\times 1} \quad \ldots \quad x^{(m)}_{3 \times 1} \right] \\
\text{i.e. }A^{[0]}_{3 \times m} &= X = \displaystyle \left[ a^{[0](1)} \quad \ldots \quad a^{[0](m)} \right]
\end{aligned}
</script>

Yup, that's right. We now have square brackets AND parantheses. What a wonderful time to be alive! 

Traversing through to layer 1, we get

<script type="math/tex; mode=display">
\begin{aligned}
Z^{[1]}_{4 \times m} &= W^{[1]^T}_{4 \times 3} A^{[0]}_{3 \times m} + b^{[1]}_{4 \times 1} \\
                    &= \displaystyle \left[ z^{[1](1)}_{4\times 1} \quad \ldots \quad z^{[1](m)}_{4\times 1} \right] \\
A^{[1]}_{4 \times m} &= g^{[1]}(Z^{[1]}) \\
                    &= \displaystyle \left[ a^{[1](1)}_{4\times 1} \quad \ldots \quad a^{[1](m)}_{4\times 1} \right]
\end{aligned}
</script>

And then onto layer 2 (the output layer, <mark>our predictions</mark>), our matrices are updated as:

<script type="math/tex; mode=display">
\begin{aligned}
Z^{[2]}_{m \times 1} &= W^{[2]^T}_{1 \times 4} A^{[1]}_{4 \times m} + b^{[2]}_{1 \times 1} \\
\hat{Y}_{m \times 1} &= A^{[2]} = g^{[2]}(Z^{[2]})
\end{aligned}
</script>

## Gradient Descent

Phew. That's our problem and the network defined. Now we're ready to do our gradient descent. If you need a refresher, look [here][week-2-part-3] where we did this for a single neuron. Thankfully, differentiation is linear, and our derivatives are linearly independent (if you don't care what those terms mean, you can still be a great deep learning guy, don't worry!), what applies to one neuron easily extends to the full set. The basic steps are:

1. Initialize <script type="math/tex"> W, b </script> 
2. Find updates through forward propagation and backpropagation
3. Repeat step 2 till convergence. Simple?

Let's get started then?

### Initialisation

For [our problem in the previous post][week-2-part-3] involving a single neuron and logistic regression, we said it was fine to initialize all variables to zero. I mean there's a single neuron, and it can learn anyway.  Here, we have *multiple neurons* (in fact a whole network!), and initialising them to zero won't work. Even initialising all of them to the same value won't work. 

Any guesses why?

I'm not sure if you got that right or wrong, so I'm just going to tell you. It's because, if you initialise all of them to the same value, they will all be computing the same function as the "signal passes through the network". What that means is that instead of a whole layer, you might just have one big neuron! Cool logic, eh?

Instead, we initialise all these things to small values, between 0 and 1 usually. They're small because some of our choices for activation functions have nice non-zero values for their derivatives close to zero.

### Forward Propagation

We've done this before already in this very post, so I'll just write out the equations. The only change here will be that the <mark>activation function in the final layer will always be the sigmoid <script type="math/tex"> \sigma(z)</script></mark>, while the other activation functions are up to us. The reasoning for this will be explained later (or not all, I haven't decided yet!).

So, the forward propagation update is:

<script type="math/tex; mode=display">
\begin{aligned}
Z^{[1]} &= W^{[1]} A^{[0]} + b^{[1]} \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= g^{[2]}(Z^{[2]})   \\
        &= \sigma(Z^{[2]})
\end{aligned}
</script>

## Back Propagation

The math involved in calculating the derivatives is very complicated. I know because Andrew Ng said so! But also, I remember it being a pain when I learnt neural networks in graduate school. It's actually *very interesting* to get into matrix calculus, but maybe I'll do it in a post of its own. Here are the back-propagation update rules, <mark>written for the second layer first and then the first layer</mark> because hey, we're going backwards!

<script type="math/tex; mode=display">
\begin{aligned}
dZ^{[2]} &= A^{[2]} - Y \text { where } Y_{m \times 1} \text{ are the training labels } \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]}A^{[1]^T} \\
db^{[2]} &= \frac{1}{m} \sum dZ^{[2]} \\
dZ^{[1]} &= \left( W^{[2]^T}dZ^{[2]} \right) \cdot \left( g^{\prime[2]}(Z^{[1]}) \right) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]}A^{[0]^T}   \\
db^{[1]} &= \frac{1}{m} dZ^{[1]}
\end{aligned}
</script>

There you have it, the six commandments of back-propagation -- some crazy math and many PhDs have gone into producing those equations. Phew!

## Summary
I have to say, this is a major achievement. You really should pat yourself on the back for this. We have the general update rules for a single-layer neural network. And even looking at it, you should be able to see that extending this to multiple layers *will not be hard*. There's a certain symmetry about the rules of update, eh?

Next post, we'll go into actual deep neural networks! Yay! And after that some code, hopefully.

For now, if you see any errors here, please leave a comment and I'll correct it promptly. If you have any questions, also leave a comment and I'll answer to the best of my abilities!

[week-2-part-3]: {{ site.baseurl }}{% post_url 2017-08-19-week-2-part-3-optimise %}
[week-2-part-1]: {{ site.baseurl }}{% post_url 2017-08-12-week-2-logistic-regression-and-neural-networks-1 %}
