# Understanding Backpropagation

Now, let's delve into the mechanics of backpropagation, a fundamental process in training neural networks through gradient descent.

In essence, what we're endeavoring to compute is the gradient of the loss function concerning each weight and bias parameter. For a softmax classifier, we employ the **cross-entropy loss function**:

$$J(\hat{y}, y) = -\sum_{i=0}^{c} y_i \log(\hat{y}_i)$$

Here, $\hat{y}$​ represents our prediction vector, typically structured as:

$$\begin{bmatrix} 0.01 \ 0.02 \ 0.05 \ 0.02 \ 0.80 \ 0.01 \ 0.01 \ 0.00 \ 0.01 \ 0.07 \ \end{bmatrix}$$

Meanwhile, $y$ denotes the one-hot encoding of the correct label for the given training example. If, for instance, the label for a training instance is 4, the one-hot encoding of $y$ would appear as:

$$\begin{bmatrix} 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ \end{bmatrix}$$


Notably, in our summation $$\sum_{i=0}^{c} y_i \log(\hat{y}_i),$$ $y_i = 0$ for all $i$ except the correct label. Therefore, the loss for a specific example simplifies to the negative logarithm of the predicted probability associated with the correct classification. In our provided example, $$J(\hat{y}, y) = -\log(y_4) = -\log(0.80) \approx 0.097.$$ Evidently, as the prediction probability approaches 1, the loss tends to 0, while as it tends to 0, the loss diverges towards $+\infty$. Minimizing the cost function is pivotal for enhancing model accuracy. This is achieved by iteratively adjusting the parameters using the derivatives of the loss function with respect to each parameter:

$$
W^{[1]} := W^{[1]} - \alpha \frac{\delta J}{\delta W^{[1]}} \\ 
b^{[1]} := b^{[1]} - \alpha \frac{\delta J}{\delta b^{[1]}} \\ 
W^{[2]} := W^{[2]} - \alpha \frac{\delta J}{\delta W^{[2]}} \\ 
b^{[2]} := b^{[2]} - \alpha \frac{\delta J}{\delta b^{[2]}}
$$

Our aim during backpropagation is to compute $$\frac{\delta J}{\delta W^{[1]}},\frac{\delta J}{\delta b^{[1]}},\frac{\delta J}{\delta W^{[2]}},$$ and $$\frac{\delta J}{\delta b^{[2]}}.$$ For brevity, we denote these derivatives as $$dW^{[1]}, db^{[1]}, dW^{[2]},$$ and $$db^{[2]}.$$ We derive these values by traversing backward through the network, commencing with the calculation of $$\frac{\delta J}{\delta A^{[2]}},$$ or equivalently, $$dA^{[2]}.$$ This derivative is elegantly expressed as:

$$dA^{[2]} = Y - A^{[2]}$$

---
Verification of this derivative can be conducted via calculus:
- Starting with the definition of the cost function, in the case of cross-entropy loss, we have: $$J(\hat{y}, y) = -\sum_{i=0}^{c} y_i \log(\hat{y}_i)$$
- Where $y$ is the vector of correct labels (in one-hot format) and $\hat{y}$​ is the vector of model predictions.
- Considering $\hat{y}$​ is obtained through the last layer of the neural network, we can represent $\hat{y}$​ as $A^{[2]}$, while $y$ is represented as $Y$. Substituting $\hat{y}$​ with $A^{[2]}$ and $y$ with $Y$, we get: $$J(A^{[2]}, Y) = -\sum_{i=0}^{c} Y_i \log(A^{[2]}_i)$$
- Now, to compute $\frac{\delta J}{\delta A^{[2]}}$, let's consider a single component $i$: $$\frac{\delta J}{\delta A^{[2]}_i} = -\frac{\delta}{\delta A^{[2]}_i}$$

Verification of this derivative can be conducted via calculus:
- Starting with the definition of the cost function, in the case of cross-entropy loss, we have: $$J(\hat{y}, y) = -\sum_{i=0}^{c} y_i \log(\hat{y}_i)$$
- Where $y$ is the vector of correct labels (in one-hot format) and $\hat{y}$ is the vector of model predictions.
- Considering $\hat{y}$ is obtained through the last layer of the neural network, we can represent $\hat{y}$ as $A^{[2]}$, while $y$ is represented as $Y$. Substituting $\hat{y}$ with $A^{[2]}$ and $y$ with $Y$, we get: $$J(A^{[2]}, Y) = -\sum_{i=0}^{c} Y_i \log(A^{[2]}_i)$$
- Now, to compute $\frac{\delta J}{\delta A^{[2]}}$, let's consider a single component $i$: $$\frac{\delta J}{\delta A^{[2]}_i} = -\frac{\delta}{\delta A^{[2]}_i} (Y_i \log(A^{[2]}_i))$$
- Applying the product rule and noting that $Y_i$ is either zero or one, hence constant with respect to $A^{[2]}_i$, we get: $$\frac{\delta J}{\delta A^{[2]}_i} = -\frac{Y_i}{A^{[2]}_i}$$
- Extending this calculation to all components, we get: $$\frac{\delta J}{\delta A^{[2]}} = -\frac{Y}{A^{[2]}}$$
- Therefore, $dA^{[2]}$ will simply be the difference between $Y$ and $A^{[2]}$, since we have a vector subtraction: $$dA^{[2]} = Y - A^{[2]}$$
- This result is crucial for computing gradients during the error backpropagation phase.

---

With $dA^{[2]}$, in hand, we proceed to compute $dW^{[2]}$ and $db^{[1]}$:

$$
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T} \\
dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}
$$

Subsequently, to derive $dW^{[1]}$ and $db^{[1]}$, we first determine $dZ^{[1]}$:

$$
dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (Z^{[1]})
$$

While a detailed mathematical explanation isn't provided here, intuition can be gleaned by examining the variables. Essentially, we apply $W^{[2]T}$ to $dZ^{[2]}$, effectively reversing the weight application between layers 1 and 2. This is followed by element-wise multiplication with the derivative of the activation function, effectively 'undoing' it to acquire precise error values.

Given that our activation function is ReLU, its derivative is relatively straightforward.

When the input value exceeds 0, the activation function assumes linearity with a derivative of 1. Conversely, when the input is negative, the activation function exhibits horizontality with a derivative of 0. Thus, $$g^{[1]\prime}(Z^{[1]})$$ manifests as a matrix of 1s and 0s contingent on the values of $Z^{[1]}$.

From this point, we proceed with analogous calculations to obtain $dW^{[1]}$ and $db^{[1]}$, using $X$ instead of $A^{[1]}$:

$$
dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T \\
dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}
$$

Subsequently, with all requisite derivatives determined, we execute parameter updates:

$$
W^{[2]} := W^{[2]} - \alpha dW^{[2]} \\
b^{[2]} := b^{[2]} - \alpha db^{[2]} \\
W^{[1]} := W^{[1]} - \alpha dW^{[1]} \\
b^{[1]} := b^{[1]} - \alpha db^{[1]}
$$

Here, $\alpha$ represents our learning rate, a 'hyperparameter' subject to our discretion. Unlike other parameters, such as the network's layer count or the units within each layer, $\alpha$ is a value we select for our model rather than one optimized by gradient descent.

In summary, we've navigated through the essential mathematics underpinning gradient descent and neural network training. To reiterate: firstly, forward propagation yields predictions from input data:

$$
Z^{[1]} = W^{[1]} X + b^{[1]} \\
A^{[1]} = g_{\text{ReLU}}(Z^{[1]}) \\
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} = g_{\text{softmax}}(Z^{[2]})
$$

Subsequently, backpropagation facilitates the computation of loss function derivatives:

$$
dZ^{[2]} = A^{[2]} - Y \\
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T} \\
dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}\\
dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]}) \\
dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T} \\
dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}
$$

Finally, we update our parameters accordingly:

$$
W^{[2]} := W^{[2]} - \alpha dW^{[2]} \\
b^{[2]} := b^{[2]} - \alpha db^{[2]} \\
W^{[1]} := W^{[1]} - \alpha dW^{[1]} \\
b^{[1]} := b^{[1]} - \alpha db^{[1]}
$$

This iterative process continues until we're content with our model's performance, with the specific iteration count being another parameter under our control.
