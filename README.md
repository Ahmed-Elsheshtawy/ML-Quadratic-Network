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
z_1 &= w_1 \cdot x + b_1 \\
a_1 &= \text{sigmoid}(z_1) \\
z_2 &= w_2 \cdot a_1 + b_2 \\
a_2 &= z_2 \text{ (linear activation)} \\
\hat{y} &= w_3 \cdot a_2 + b_3
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
L = (\hat{y} - y)^2
$$

### **Backpropagation Gradients**

$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_3} &= \frac{\partial L}{\partial \hat{y}} \cdot a_2 \\
\frac{\partial L}{\partial b_3} &= \frac{\partial L}{\partial \hat{y}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_2} &= \frac{\partial L}{\partial \hat{y}} \cdot w_3 \cdot a_1 \\
\frac{\partial L}{\partial b_2} &= \frac{\partial L}{\partial \hat{y}} \cdot w_3
\end{align*}
$$

**Sigmoid Derivative:**

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

$$
\begin{align*}
\frac{\partial L}{\partial w_1} &= \frac{\partial L}{\partial \hat{y}} \cdot w_3 \cdot w_2 \cdot x \cdot \sigma'(z_1) \\
\frac{\partial L}{\partial b_1} &= \frac{\partial L}{\partial \hat{y}} \cdot w_3 \cdot w_2 \cdot \sigma'(z_1)
\end{align*}
$$

### **Weight Update Rule**

$$
\begin{align*}
w &\leftarrow w - \alpha \cdot \frac{\partial L}{\partial w} \\
b &\leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}
\end{align*}
$$

where $\alpha$ is the learning rate.

## **Experiment Results**

### **Performance**
- **Training Loss**: ~1070 (final epoch)
- **Test Loss**: ~13,856
- **Predicted Outputs**: Network outputs a near-constant value (~36.67) for all inputs

### **Key Observation**
The network **failed to learn** the quadratic function and instead memorized a single constant value. This demonstrates **underfitting** due to insufficient model capacity.

## **Lessons Learned**

### **1. Model Capacity Matters**
A single sigmoid neuron can only learn **monotonic functions** (always increasing or decreasing). The quadratic function f(x) = x² is **non-monotonic** (decreases then increases), requiring multiple neurons to approximate.

### **2. Numerical Stability is Critical**
Several overflow errors were encountered and fixed:
- **Sigmoid overflow**: Large inputs cause e^x to overflow → Fixed with conditional computation
- **Gradient explosion**: Large gradients cause weight divergence → Fixed with gradient clipping
- **Loss overflow**: Extreme predictions cause loss calculation to fail → Fixed with output clipping

### **3. Proper Initialization is Essential**
- Initial weights in range [-100, 100] caused immediate saturation → No learning
- Weights in range [-1, 1] allowed gradients to flow → Learning possible (but limited by architecture)

### **4. Learning Rate Tuning**
- Too small (0.0001): Extremely slow convergence
- Too large (0.01): Weight divergence and NaN values
- Balanced (0.001): Stable training without numerical issues

## **Why the Network Failed**

The 1-neuron architecture cannot represent $x^2$ because:
1. **Symmetry Problem**: $f(-x) = f(x)$, but a single sigmoid path cannot learn this
2. **Non-Monotonicity**: $x^2$ has a minimum at $x=0$, requiring multiple neurons to approximate
3. **Local Minimum**: The network found that predicting a constant (~36.67) minimizes average loss better than any function it can represent

## **Next Steps**
To successfully learn $x^2$, the network would need:
- **More hidden neurons** (3-5 minimum) to increase representational capacity
- **Multiple hidden layers** for more complex function approximation
- **Better activation functions** (ReLU, tanh) that might handle non-linearity better

## **Technical Challenges Solved**
1. ✅ Overflow in sigmoid function
2. ✅ Overflow in gradient calculation (exponential terms)
3. ✅ Overflow in loss calculation
4. ✅ NaN propagation from exploding gradients
5. ✅ Weight initialization causing vanishing gradients

## **Conclusion**
This experiment demonstrates that **neural network architecture must match problem complexity**. A 1-neuron network is fundamentally incapable of learning $x^2$, regardless of training time or hyperparameter tuning. This hands-on experience reinforces the importance of the **universal approximation theorem** in theory versus practical learning in practice.