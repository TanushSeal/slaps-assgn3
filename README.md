# SLAPS-Assignment 3: Visual Arithmetic

## 1. Executive Summary

This report analyzes the performance of two distinct AI methodologies applied to the "MNIST Arithmetic" problem. The goal was to build a system capable of recognizing handwritten digits and performing arithmetic operations (Summation, Multiplication, Subtraction).

The experiment compared:

1. **Baseline CNN:** A pure deep learning approach mapping image pairs directly to sum labels.

2. **Neuro-Symbolic (DeepProbLog):** A hybrid approach combining neural perception with symbolic logic rules.

**Key Finding:** The Neuro-Symbolic model significantly outperformed the Baseline (**91.50% vs 77.58%**) and demonstrated **Zero-Shot Transfer** capabilities, solving multiplication and subtraction problems without explicit training on those tasks.

## 2. Methodology

### 2.1. Task Definition

* **Input:** Pairs of MNIST images ($28 \times 28$ grayscale).

* **Primary Task:** Predict the sum of the digits represented in the images.

* **Secondary Task (Generalization):** Predict the product and difference of the digits without retraining.

### 2.2. Method A: Baseline CNN (Connectionist)

* **File:** `baseline_cnn.py`

* **Architecture:** A Siamese-style Convolutional Neural Network.

  * **Feature Extraction:** Two streams (sharing weights) process the input images. Each stream consists of 2 Convolutional layers (Kernel $5 \times 5$) followed by Max Pooling.

  * **Fusion:** Feature vectors are flattened and concatenated.

  * **Classification:** A fully connected network ($128 \times 2 \to 19$) maps the fused features to one of 19 classes (Sums 0–18).

* **Training Strategy:** Treated arithmetic as a classification problem. The model attempts to learn visual patterns corresponding to specific sums (e.g., memorizing that a specific pixel pattern of "3" and "4" maps to class "7").

### 2.3. Method B: Neuro-Symbolic (DeepProbLog)

* **File:** `pro.py`

* **Architecture:** Decoupled Perception and Reasoning.

  * **Perception (Neural):** A standard CNN (`MNISTNet`) maps a single image to a probability distribution over digits 0–9.

  * **Reasoning (Symbolic):** Probabilistic logic rules defined in Prolog.

* **Logic Rules:**

  * **Sum:** `sum_digits(I1, I2, S) :- digit(I1, N1), digit(I2, N2), S is N1 + N2.`

  * **Product (Extension):** `product_digits(I1, I2, P) :- digit(I1, N1), digit(I2, N2), P is N1 * N2.`

  * **Difference (Extension):** `diff_digits(I1, I2, D) :- digit(I1, N1), digit(I2, N2), D is N1 - N2.`

* **Training Strategy:** The model is trained end-to-end using the Sum task. Gradients from logical inconsistencies (predicting the wrong sum) are backpropagated to correct the digit classifier.

## 3. Experimental Results

Analysis of the provided `log.txt` reveals distinct performance trajectories for both models.

### 3.1. Training Convergence

* **Baseline CNN:**

  * **Trajectory:** Loss decreased rapidly in early epochs but plateaued around **0.56–0.58** by Epoch 20.

  * **Observation:** The aggressive learning rate decay (seen in logs dropping to `1.0e-05` quickly) likely caused the model to settle into a suboptimal local minimum. The model struggled to generalize the concept of addition, relying instead on memorizing specific pair configurations.

* **Neuro-Symbolic:**

  * **Trajectory:** Started with high loss (~2.48) but converged to near-zero (**~0.01**) by iteration 1500.

  * **Observation:** Convergence was stable and efficient. Because the logic component handles the arithmetic exactly, the neural network only needed to learn the simpler task of 10-class digit classification, rather than 100-class pair combinations.

### 3.2. Quantitative Accuracy

The following results were extracted directly from the final evaluation steps in `log.txt`.

| Model | Task | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | Sum (Test Set) | **77.58%** | Trained |
| **Neuro-Symbolic** | Sum (Test Set) | **91.50%** | Trained |
| **Neuro-Symbolic** | Product | **93.00%** | **Zero-Shot** (Untrained) |
| **Neuro-Symbolic** | Difference | **91.50%** | **Zero-Shot** (Untrained) |

## 4. Discussion & Analysis

### 4.1. The Failure of the Baseline

The Baseline model achieved only 77.58% accuracy. This performance gap highlights the **Combinatorial Complexity** problem.

* The Baseline sees "Image(2) + Image(3)" and "Image(4) + Image(1)" as completely different visual inputs that happen to share label "5".

* It lacks the semantic understanding that the label "5" is a mathematical result. It tries to memorize pairs. With only 50 epochs and a simple architecture, it could not memorize the visual variance of all $10 \times 10$ combinations effectively.

### 4.2. The Success of Semantic Grounding

The Neuro-Symbolic model achieved 91.50% accuracy on Sums.

* By explicitly programming the rule $N_1 + N_2 = S$, the system breaks the problem down. The neural network is grounded: it is forced to learn "What does a 2 look like?" rather than "What does a 2+3 pair look like?".

* This reduces the complexity from $O(N^2)$ pairs to $O(N)$ digits.

### 4.3. Zero-Shot Generalization (The "Extension")

The most critical advantage demonstrated in this experiment is **Zero-Shot Transfer**.

* The `pro.py` file defined rules for `product_digits` and `diff_digits`.

* The logs show that despite **never training on multiplication or subtraction labels**, the model achieved **93.00%** and **91.50%** accuracy on these tasks respectively.

* **Implication:** This proves the model learned the *concepts* (the digits), allowing it to reason about *any* relationship involving those concepts. A standard CNN would require a new output layer and full retraining to switch from Addition to Multiplication.

## 5. Conclusion

The experiment confirms that while standard Deep Learning (Baseline) can approximate arithmetic functions, it is inefficient and brittle (77.58% accuracy) when training data is limited. The Neuro-Symbolic approach (DeepProbLog) successfully integrated perception and reasoning, achieving superior accuracy (91.50%) and, crucially, the ability to generalize to new arithmetic tasks without additional training.
"""

