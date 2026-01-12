# **ML Quadratic Equation Approximation**

## **Overview**
This project is a self-study experiment to understand the fundamentals of neural networks by building a simple feedforward network from scratch (no frameworks) to approximate the quadratic function $f(x) = x^2$.

The goal is to explore:
- How neural network architecture affects learning capacity
- The importance of proper weight initialization
- Numerical stability challenges in backpropagation
- The bias-variance tradeoff in practice

## **Experiment Setup**

### **Network Architecture**
- **Input Layer**: 1 neuron (input x)
- **Hidden Layer**: 1 neuron with sigmoid activation
- **Hidden Layer**: 1 neuron with linear activation
- **Output Layer**: 1 neuron (predicted y)

### **Training Data**
- **Training Inputs**: $x \in \{-10, -9, -8, \ldots, 8, 9, 10\}$
- **Training Outputs**: $y = x^2$
- **Testing Inputs**: $x \in \{-15, -12, -11, -0.5, 0.5, 11, 12, 15\}$

### **Hyperparameters**
- **Learning Rate**: 0.001
- **Epochs**: 1000
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)

## **Key Equations**

### **Forward Propagation**

$$
\begin{align*}
n_{1}\mathrm{Input} &= w_1 \cdot x + b_1 \\
n_{1}\mathrm{Output} &= \text{sigmoid}(n_{1}\mathrm{Input}) \\
n_{2}\mathrm{Input} &= w_2 \cdot n_{1}\mathrm{Output} + b_2 \\
n_{2}\mathrm{Output} &= n_{2}\mathrm{Input} \text{ <-- (linear activation)} \\
\mathrm{output} &= w_3 \cdot n_{2}\mathrm{Output} + b_3
\end{align*}
$$

### **Sigmoid Activation (Numerically Stable)**

$$
\sigma(x) = 
\begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{1 + e^x} & \text{if } x < 0
\end{cases}
$$

### **Loss Function**

$$
L = (\text{predicted} - \text{correct})^2
$$

### **Backpropagation Gradients**

$$
\frac{\partial L}{\partial \text{output}} = 2(\text{predicted} - \text{correct})
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_3} &= \frac{\partial L}{\partial \text{output}} \cdot n_{2}\mathrm{Output} \\
\frac{\partial L}{\partial b_3} &= \frac{\partial L}{\partial \text{output}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_2} &= \frac{\partial L}{\partial \text{output}} \cdot w_3 \cdot n_{1}\mathrm{Output} \\
\frac{\partial L}{\partial b_2} &= \frac{\partial L}{\partial \text{output}} \cdot w_3
\end{align*}
$$

**Sigmoid Derivative:**

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_1} &= \frac{\partial L}{\partial \text{output}} \cdot w_3 \cdot w_2 \cdot x \cdot \sigma'(n_{1}\mathrm{Input}) \\
\frac{\partial L}{\partial b_1} &= \frac{\partial L}{\partial \text{output}} \cdot w_3 \cdot w_2 \cdot \sigma'(n_{1}\mathrm{Input})
\end{align*}
$$

### **Weight Update Rule**

$$
\begin{align*}
w &\leftarrow w - \mathrm{learningRate} \cdot \mathrm{gradient} \\
b &\leftarrow b - \mathrm{learningRate} \cdot \mathrm{gradient}
\end{align*}
$$

## **Results**

### **Test Loss**
```
Test Loss: 13856.675664549257
```
### **Correct Outputs**
```
Correct Outputs: 
[225, 144, 121, 0.25, 0.25, 121, 144, 225]
```
### **Predicted Outputs**
```
Predicted Outputs: 
[36.66880718269546, 36.66880718269546, 36.66880718269546, 36.66880718270334, 36.66880718271128, 36.668807206517144, 36.66880723051466, 36.66880756944144]
```

The network achieved training loss of ~1070 and test loss of ~13,856, outputting a near-constant value (~36.67) for all inputs. This demonstrates complete failure to learn due to insufficient model capacity.

## **Key Findings**

**Model Capacity**: A single sigmoid neuron cannot learn non-monotonic functions like $x^2$. The network converged to a local minimum where predicting a constant minimizes average loss.

**Numerical Stability**: Fixed overflow errors in sigmoid calculation, gradient computation, and loss calculation through conditional computation and gradient clipping.

**Initialization and Learning Rate**: Large initial weights [-100, 100] caused saturation. Small weights [-1, 1] with learning rate 0.001 enabled stable training but couldn't overcome architectural limitations.

## **Challenges Addressed**
- Overflow in sigmoid function from large exponents
- Gradient explosion during backpropagation
- NaN propagation from numerical instabilities
- Weight initialization causing vanishing gradients

## **Conclusion**
This experiment demonstrates that neural network architecture must match problem complexity. A 1-neuron network fundamentally cannot learn $x^2$ regardless of training time or hyperparameter tuning. Successfully learning this function requires multiple hidden neurons to approximate the non-monotonic behavior.