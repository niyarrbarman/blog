---
layout: post
title: "Neural Networks 101"
date: 2026-06-22 00:00:00 +0000
categories: [learning]
tags: [ml, gradient, backprop, optimization]
author: niyar r barman
excerpt: "learning log of neural networks for future j*b interviews"
---

this is inspired from [Alisa's book of LLMs](https://alisawuffles.notion.site/alisa-s-book-of-llms) to keep a track of all of the stuff i cover and on the off-chance that it might help someone. this blog covers everything neural networks. 

<div class="contents-index" markdown="1">

* toc 
{:toc}

</div>


## gradients
gradients are basically how neural nets figure out which way to move. when we train a neural net, we start with random weights. the model makes a prediction, we calculate a loss and then we ask: *if i slightly change this weight, how does the loss change?* the answer to that is given by the gradient.

if the gradient is positive, increaing the weight increases the loss, so we probably want to decrease the weight. if the gradient is negative, increasing the weight decreases the loss, so we probably want to increase the weight. 

this is the intuition behind gradient descent.

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

### partial derivatives

for a neural net, the loss depends on a lot of parameters: $$ L = L(w_1, w_2, \ldots, w_n)$$. so we do not ask how the loss changes with respect to one weight but with respect to every single weight.

$$\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_n} $$

each of these is a partial derivative. a partial derivative tells us how the output changes with respect to one variable while all the others are fixed. this is especially useful as neural nets have millions, billions (or now TRILLIONS!!) of weights and we need to know how each of them contributed to the final loss.

### gradient of a scalar

if we have a scaler function $$f:\R^n \rightarrow \R$$, then the gradient of $$f$$ with respect to $$x$$ is:

$$
\nabla_x f = 
\left [
    \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}
\right]
$$

the gradient points in the direction of the steepest increase of the function but we want to minimize our loss, so we move in the oppisite direction:

$$
x \leftarrow x - \eta \nabla_x f
$$

### jacobian matrix
the basic building block of vectorized gradients is the *Jacobian Matrix*. if we have a function $$f: \R^n \rightarrow \R^m$$ that maps a vector of length *n* to a vector of length *m* : 

$$
f(x) = [f_1(x_1, \ldots, x_n), f_2(x_1, \ldots, x_n),\ldots, f_m(x_1, \ldots, x_n)]
$$ 

its jacobian is the following $$m \times n$$ matrix:

$$
\frac{\partial f}{\partial x} 
= 
\left[
\begin{array}{ccc}
\frac{\partial f_1}{\partial x_1} & \ldots & \frac{\partial f_1}{\partial x_n} \\ 
\vdots & \ddots & \vdots \\ 
\frac{\partial f_m}{\partial x_1} & \ldots & \frac{\partial f_m}{\partial x_n} 
\end{array}
\right]
$$

each row tells us how one output changes with respect to all the inputs. each column tells us how all outputs with respect to one input. 

a gradient is what we get when the function outputs a scalar $$f: \R^n \rightarrow \R$$. a jacobian is what we get when the function outputs a vector $$f: \R^n \rightarrow \R^m$$

### some useful derivatives

#### sigmoid
the sigmoid function is 

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

now, 

$$
\begin{aligned}
\frac{d}{dx}(\sigma(x))&=\frac{d}{dx}(1 + e^{-x})^{-1} \\
&= -1 \times (1 + e^{-x})^{-2} \times e^{-x} \times -1 \\
&= \frac{e^{-x}}{(1 + e^{-x})^{-2}} \\
&= \sigma(x)(1 - \sigma(x))
\end{aligned}
$$


so the derivative can be written using the sigmoid output itself. also sigmoid saturates when $$x$$ is very positive or very negative. in both cases the derivative becomes very small which can cause [*vanishing gradients*](#vanishing-and-exploding-gradients). 

#### tanh

$$
\tanh(x) =
\frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

its derivative is:

$$
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
$$

tanh is zero-centered unlike sigmoid, but it can still saturate for large positive or negative values.

#### relu

$$
\text{ReLU}(x) = \max(0, x)
$$

its derivative is: 

$$
\frac{d}{dx} \text{ReLU(x)} =
\begin{cases}
1, & x > 0 \\
0, & x < 0
\end{cases}
$$

at $$x=0$$, it is technically not differentiable, but in practice frameworks just choose some convention. relu is simple and works well but if a neuron keeps getting negative inputs, its gradient becomes zero. this is the *dead relu problem*.

#### swish
swish also called silu in pytorch is :

$$
\text{swish}(x) = x\sigma(x)
$$

its derivative is:

$$
\begin{aligned}
\frac{\partial}{\partial x}\operatorname{swish}(x)
&= \frac{\partial}{\partial x}\left(x \cdot \sigma(x)\right) \\
&= \sigma(x) + x \cdot \sigma'(x) \\
&= \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) \\
&= \sigma(x) + x\sigma(x) - x\sigma(x)^2 \\
&= \sigma(x) + \operatorname{swish}(x)(1 - \sigma(x))
\end{aligned}
$$

swish is smooth unlike relu and sigmoid. it does not squash everything between 0 and 1 becuase the input $$x$$ is still multiplied outside. 

#### softmax
softmax takes a vector of logits and turns it into a probability distribution.

for a vector $$z = [z_1, z_2, \ldots, z_j]$$, softmax is defined as:

$$
p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

softmax is interesting because each output depends on all inputs and not just one input. so its derivative is not a simple scalar derivative. its a jacobian.

the softmax derivative is:

$$p_i(\delta_{ij} - p_i)$$

where $$\delta_{ij}$$ is 1 when $$i = j$$ and 0 otherwise.


$$
\frac{\partial p_i}{\partial z_j} = 
\begin{cases}
p_i(1 - p_i) & i = j\\
-p_i p_j & i \neq j
\end{cases}
$$

conceptually, increasing one logit increases its own probability, but decreases the probabilities of the other classes because all probabilities must still sum to 1.

#### softmax with CE
in language models, softmax is followed by CE loss. if the class/token is $$t$$m then the loss is: $$L = \log p_t$$. so the loss only directly depends on the probability assigned to the correct class.

the derivative of the loss with respect to probabilities is:

$$
\frac{\partial L}{\partial p_i} =
\begin{cases}
-\frac{1}{p_t} & i = t \\
0 & i \neq t
\end{cases}
$$

now if we want the derivative with respect to the logits, we can use the chain rule:

$$
\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial z_j}
$$

now most of the terms vanish because $$\frac{\partial L}{\partial p_i}$$ is non-zero only for the correct class $$t$$.

$$
\begin{aligned}
\frac{\partial L}{\partial z_j} 
&=\frac{\partial L}{\partial p_t} \frac{\partial p_t}{\partial z_j} \\
&=\begin{bmatrix}0&\cdots&-\frac{1}{p_t}&\cdots &0\end{bmatrix}\begin{bmatrix}\frac{\partial p_1}{\partial z_1}&\cdots&\frac{\partial p_1}{\partial z_{\mathcal V}}\\&\ddots&\\\frac{\partial p_{\mathcal V}}{\partial z_1}&\cdots&\frac{\partial p_{\mathcal V}}{\partial z_{\mathcal V}}\end{bmatrix}\\
&=-\frac{1}{p_t}\frac{\partial p_t}{\partial z_i}\in\mathbb R^{\mathcal V}
\end{aligned}
$$

therefore:

$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
&= \begin{cases}
\frac{-1}{p_j} p_j(1-p_j) & j = t \\
\frac{-1}{p_t} (-p_tp_j) & j \neq t
\end{cases} \\
&= \begin{cases}
p_j - 1 & j = t \\
p_j & j \neq t
\end{cases}
\end{aligned}
$$

or in vector form $$p-y$$ where $$y$$ is the one-hot target vector.

softmax alone has a full jacobian, but softmax $$+$$ CE simplifies to $$p-y$$

this is why the gradient at the logits is easy to interpret:
- for the correct token, the gradient is $$p_t - 1$$
- for the incorrect token, the gradient is $$p_j$$

during gradient descent, this increases the correct logit and decreases the wrong logits. so the model is basically pushed towards assigning more probability to the correct token.

### hessians

the gradient tells us the slope, the hessian tells us about the curvature. if the gradient is made of first derivatives, the hessian is made of second derivatives. for a scalar function $$f: \R^n \rightarrow \R$$, the hessian is:

$$
H =
\left[
\begin{array}{ccc}
\frac{\partial^2 f}{\partial x_1^2} & \ldots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \ldots & \frac{\partial^2 f}{\partial x_n^2}
\end{array}
\right]
$$

in deep learning, the full hessian is usually too expensive to compute but conceptually, it is useful as it tells us about the local shape of the loss landscape. i will come back to properly when i study about [hessians/curvature in optimization](#hessians--curvature)

## backprop

backpropagation is the algorithm used to compute gradients in a neural net. the important thing is that a neural net is not treated as one huge function whose derivative we manually write out. so instead the forward pass is broken down into many smaller operations, and these operations form a computation graph. 

for example a model might look like:

$$
x \rightarrow h_1 \rightarrow h_2 \rightarrow \hat{y} \rightarrow L
$$

backprop computes $$\frac{\partial L}{\partial \theta}$$ for every trainable parameter $$\theta$$ in the model. the core idea is simple. each node receives a gradient from the node after it, multiplies it by its own local gradient, and passes it backward.

### compuation graph

a computation graph represents a function as small operations connected together. for example:

$$
f(x, y, z) = (x + y)\max(y, z)
$$

we can split this into smaller nodes:

$$a = x+y \quad  b=\max(y, z) \quad   f=ab$$

so the forward pass is:

$$x, y, z \rightarrow a, b \rightarrow f$$

this is useful because each small operation has a simple local derivative. the whole derivative is then built by combining these local derivatives using the chain rule.

### local gradient, upstream gradient, downstream gradient

for every node in the graph, there are two important gradients. the *local gradient* tells us how the node output changes with respect to its own input. the *upstream gradient* is the gradient coming from the later part of the graph. then the node sends back a *downstream gradient* to the earlier nodes. so roughly : $$\text{upstream gradient} \times \text{local gradient}$$

for example if:

$$
z = f(y) \quad y = g(x)
$$

then the upstream gradient is:

$$
\frac{\partial z}{\partial y}\frac{\partial y}{\partial x}
$$

and, the local gradient is:

$$
\frac{\partial y}{\partial x}
$$

<!--### example-->

### node intuitions

some operations have very simple gradient behaviour. 

#### for addition 

$$z=x+y$$

the upstream gradient is copied to both inputs.

$$
\frac{\partial z}{\partial x} = 1 \quad \frac{\partial z}{\partial y} = 1
$$

so addition distributes gradients.

#### for multiplication:

$$z=x \times y$$

the gradients switch coefficients.

$$
\frac{\partial z}{\partial x} = y \quad \frac{\partial z}{\partial y} = x 
$$

so if the upstream gradient is $$g$$:

$$
\frac{\partial z}{\partial x} = gy \quad \frac{\partial z}{\partial y} = gx 
$$

#### for max:

$$z = \max(x, y)$$

the gradient flows only to the input that won the max.

$$
\frac{\partial z}{\partial x} = 1 \quad \frac{\partial z}{\partial y} = 0 \quad \text{if} \quad x > y \\
\frac{\partial z}{\partial x} = 0 \quad \frac{\partial z}{\partial y} = 1 \quad \text{if} \quad x < y 
$$

so max routes gradients. this is also again why operations like ReLU can create zero gradients.

### forward and backward pass

training has two main phases. the forward pass computes activations and the loss:

$$
x \rightarrow \hat{y} \rightarrow L
$$

the backward pass computes the gradients

$$
L \rightarrow \hat{y} \rightarrow x
$$

during the forward pass, the model usually stores intermediate activations. these activations are needed during the backward pass.

for example, if we have $$y = xW$$. we need the input activation $$x$$. this is why training uses more memory than inference. in inference, we only need the forward pass. in training, we need the forward pass $$+$$ stored activations for backprop.

### vanishing and exploding gradients

## optimization

### hessians / curvature





# refs
- Alisa Liu. [Alisa's book of LLMs](https://alisawuffles.notion.site/alisa-s-book-of-llms)
- Kevin Clark. [Computing Neural Network Gradients](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)
