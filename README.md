# Self-Attention Mechanism Using NumPy

## Overview

This project demonstrates how the **Self-Attention Mechanism** can be implemented using **Python** and **NumPy**. Self-attention is a key component of Transformer models that enables each input token to determine the importance of other tokens when generating contextual representations.

This implementation walks through every stage of the attention process, from creating input embeddings to computing the final attention output.

---

## Objective

* Understand the basic concepts of the Self-Attention mechanism.
* Generate Query (Q), Key (K), and Value (V) matrices from input embeddings.
* Compute attention scores and normalize them using the Softmax function.
* Produce contextual representations through the attention mechanism.

---

## Technologies Used

* Python 3.x
* NumPy

---

## Input Embeddings

```python
X = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 1.0],
    [1.0, 1.0, 0.0]
])
```

The above matrix represents the embeddings of three input tokens used in the attention computation.

---

## Implementation Steps

### 1. Initialize Weight Matrices

Random weight matrices are created for:

* Query (WQ)
* Key (WK)
* Value (WV)

### 2. Generate Query, Key, and Value

The input embeddings are multiplied with their corresponding weight matrices.

```text
Q = X × WQ
K = X × WK
V = X × WV
```

### 3. Compute Attention Scores

The similarity between each Query vector and every Key vector is calculated.

```text
Attention Scores = Q × Kᵀ
```

### 4. Scale the Attention Scores

The scores are divided by the square root of the key dimension to maintain numerical stability.

```text
Scaled Scores = Scores / √dₖ
```

where **dₖ** represents the dimension of the Key vectors.

### 5. Calculate Attention Weights

The Softmax function converts the scaled scores into probability values.

```text
Attention Weights = Softmax(Scaled Scores)
```

These probabilities determine the importance assigned to each input token.

### 6. Compute the Final Attention Output

The attention weights are multiplied by the Value matrix to obtain contextualized representations.

```text
Attention Output = Attention Weights × V
```

---

## Output
<img width="421" height="680" alt="image" src="https://github.com/user-attachments/assets/f508729e-b7b3-4009-95eb-69e38896b4bf" />






## Project Structure

```text
Self-Attention/
│
├── self_attention.py
├── README.md
```

---

## Key Concepts

* Self-Attention
* Input Embeddings
* Query (Q)
* Key (K)
* Value (V)
* Attention Scores
* Scaled Dot-Product Attention
* Softmax Function
* Attention Weights
* Transformer Models

---

## Formula

```text
Attention(Q, K, V) = Softmax((QKᵀ) / √dₖ) V
```

---

## Learning Outcomes

After completing this project, you will be able to:

* Explain the working process of the Self-Attention mechanism.
* Understand the purpose of Query, Key, and Value matrices.
* Calculate attention scores and attention weights.
* Recognize the importance of scaling before applying Softmax.
* Understand how contextual representations are generated in Transformer models.

---

## Conclusion

This project provides a simple and easy-to-understand implementation of the Self-Attention mechanism using **Python** and **NumPy**. It demonstrates the complete Scaled Dot-Product Attention process and helps build a strong foundation for learning Transformer architectures and modern Large Language Models (LLMs).
