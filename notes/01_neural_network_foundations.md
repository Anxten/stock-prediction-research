# Neural Network Foundations

## 1. The Core Idea
Neural Networks are essentially function approximators. They take inputs, pass them through layers of weights and biases, and produce an output.
- **Formula:** $y = f(\sum(w_i x_i) + b)$

## 2. Backpropagation (The Calculus Part)
This is how the network learns. It minimizes the error (Loss Function) by adjusting weights.
- It uses the **Chain Rule** from Calculus: $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$
- We calculate the gradient of the Loss with respect to each weight to find the direction of steepest descent.

## 3. ReLU Activation
- **Problem with Sigmoid:** Can cause Vanishing Gradient (derivatives become near zero).
- **ReLU (Rectified Linear Unit):** $f(x) = max(0, x)$.
- **Benefit:** Computationally efficient and helps gradients flow better during backpropagation.