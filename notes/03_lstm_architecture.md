# Long Short-Term Memory (LSTM)

## 1. The Savior of Time Series
LSTM is a special kind of RNN capable of learning long-term dependencies. It was explicitly designed to solve the **Vanishing Gradient Problem** found in standard RNNs.

## 2. The Core Idea: Cell State
The key to LSTMs is the **Cell State** (the horizontal line running through the top of the diagram).
- **The Conveyor Belt:** It allows information to flow down the sequence unchanged, preserving long-term memory.
- **Additive Updates:** Unlike standard RNNs that use multiplicative updates (causing gradients to vanish), LSTMs use **Pointwise Addition**. This creates a "Gradient Superhighway," allowing gradients to flow back to early time steps without shrinking to zero.

## 3. The Three Gates (The Control Mechanism)
LSTMs use "Gates" to regulate information. Each gate uses a **Sigmoid** layer to decide "how much" information passes (0 to 1).

- **Forget Gate:** The first step. It decides what to discard from the previous Cell State.
- **Input Gate:** Uses a **Sigmoid** (to decide which values to update) and a **Tanh** (to create a vector of new candidate values scaled between -1 and 1) to update the Cell State.
- **Output Gate:** Decides what the next Hidden State should be by filtering the current Cell State through a **Sigmoid** and scaling it with **Tanh**.