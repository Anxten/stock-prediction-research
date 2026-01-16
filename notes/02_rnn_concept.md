# Recurrent Neural Networks (RNN)

## 1. The Core Difference: "Memory"
Unlike standard Feed-Forward Neural Networks (which handle independent data), RNNs are designed for **Sequential Data** (Time Series, Text, Stock Prices).
- **The Loop:** The output from the previous step is fed back into the network as input for the current step.
- **Formula:** $h_t = f(W \cdot h_{t-1} + U \cdot x_t + b)$
  *(Current State = Previous State + Current Input)*

## 2. Unrolling (Unfolding)
To train an RNN, we unroll the loop over time.
- If we have 30 days of stock data, the RNN unrolls into a 30-layer deep network.
- Crucial Concept: **Parameter Sharing**. The weights ($W, U$) are exactly the same for every time step.

## 3. The Limitation: Vanishing Gradient
When we perform **Backpropagation Through Time (BPTT)** on long sequences:
- We multiply gradients (Chain Rule) many times (e.g., 50 times for 50 days).
- If gradients are small (< 1), multiplying them repeatedly makes them approach **Zero**.
- **Result:** The network stops learning from early data points. It has "Short-term Memory" and forgets long-term trends.
- **Solution:** LSTM (Long Short-Term Memory).