
---
title: Mobilenet - A summary
author: pritamsukumar
type: post
date: 2020-05-29
draft: true
featured_image: 
url: 
categories:
---

Here, I write my notes on the MobileNet paper:

* MobileNet bucks the trend of achieving higher accuracy through deeper and more complicated networks by focusing on making an efficient network architecture.
    * Only two hyperparameters
    * Low latency
    * Low hardware requirements - suitable for mobile vision applications
    * Allows for developer to choose a model that matches the resource restrictions
* To reduce computation in first few layers, "depthwise separable convolutions [TODO]" are used. 

## Depthwise Separable Convolution

* A standard convolution is factorized into:
    * A depthwise convolution (for MobileNet, this is a single filter for each input channel), and
    * a 1x1 convolution to comine the outputs of the depthwise convolution
    * This factorization drastically reduced computation size.
{:refdef: style="text-align: center;"}
![Depthwise Separable convolutions](/assets/mobilenet_review/depthwise_separable_convolutions.png)*The idea is that (a) is replaced by (b) and (c)*
{: refdef}
* The computation is reduced quite significantly. (Sec 3.1 in the paper)
* Sec 3.2 talks about training and matrix operation optimization
* Sec 3.3 introduces the first hyperparameter: The Width Multiplier. This reduces the number of input channels at any given layer, thus reducing the computational cost.
* Sec 3.4 introduces our second hyperparameer: The resolution multiplier. This is applied to the original input image.
* Sec 4 is all showing off.