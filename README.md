# Perceptron Algorithm for Data Classification

This repository contains an implementation of the perceptron algorithm for data classification. The project covers two primary approaches:

- **Binary Perceptron:**  
  The classic single-layer perceptron algorithm is implemented for binary classification. Three distinct binary classifiers are trained—for class pairs (class 1 vs. class 2, class 2 vs. class 3, and class 1 vs. class 3)—and their performance is evaluated after 20 training iterations.

- **One-vs-Rest Multi-Class Classification:**  
  The binary perceptron is extended to handle multi-class problems using the one-vs-rest approach. Each class is assigned its own binary classifier (by marking that class as positive and the rest as negative), and overall training and test accuracies are computed.

Additionally, the project explores the impact of L2 regularization on the one-vs-rest classifier by applying different regularization coefficients and observing how model performance changes.

---

## Overview

The perceptron algorithm is a fundamental building block in neural network theory, used for finding a linear decision boundary between classes. While the single-layer perceptron works well for linearly separable problems, the multi-layer perceptron (and its variations) can handle non-linear patterns via methods like backpropagation. The primary objective of this algorithm is to identify a linear decision boundary that can effectively separate two distinct classes. This is accomplished by iteratively adjusting the weights and bias term. The perceptron employs a signed linear function as its activation function, and the learning process involves updating the weights whenever the predicted label differs from the true label. In this particular project, I undertook the task of implementing the binary perceptron algorithm entirely from the ground up.


In this project, the following aspects are addressed:

- **Training Procedure:**  
  - Initialize weights and bias to zero.
  - Iterate through the training dataset for a fixed number of iterations (20 in this case), computing an activation score for each sample.
  - Update the weights and bias when a misclassification is detected.
  - Output the fixed parameters after training.

- **Testing Procedure:**  
  - With the fixed parameters, compute the activation score for a new sample.
  - Use the sign of the computed score as the predicted class label.

- **One-vs-Rest Strategy:**  
  Each class in our multi-class dataset is treated independently by training a binary classifier for each. This approach allows the construction of a decision boundary that separates the positive class from all others.

- **L2 Regularization Experimentation:**  
  L2 regularization is applied to control the magnitude of the weights. Experiments show that a regularization coefficient of **0.01** produces the best training and test accuracies, while stronger regularization (coefficients ≥ 0.1) causes a dramatic drop in performance.

---

## Algorithm Description

### Perceptron Algorithm

The perceptron classifier is designed for binary tasks with real-valued inputs (X) and mappings (y). The training procedure follows these steps:

1. **Initialization:**  
   - Set all weights to 0.
   - Set the bias term to 0.

2. **Training Loop (for each iteration up to MaxIter):**  
   - For each sample in the training dataset, calculate an activation score using the current weights and bias.
   - If the prediction is incorrect (i.e., the product of the sample's true label and its activation score is less than or equal to 0), update the weights and bias accordingly.
   - Continue for a preset number of iterations (20 iterations in this project).

3. **Output:**  
   - Return the final weights and bias.

A pseudocode example is provided in the repository for clarity.

### One-vs-Rest Approach

To solve multi-class classification problems, a one-vs-rest strategy is employed:
- For each class, a separate perceptron is trained where that class is considered positive (+1) and all other classes negative (-1).
- This process is repeated for every class.
- Overall accuracies are computed by aggregating the predictions from each binary classifier.

### L2 Regularization

An additional experiment applies L2 regularization to the perceptron in the one-vs-rest setting. The goal is to penalize large weight values, which may help to avoid overfitting. Experimental observations include:

- **Coefficient 0.01:**  
  - Train Accuracy: 62.24%
  - Test Accuracy: 64.41%
- **Higher Coefficients (≥ 0.1):**  
  - Both training and test accuracies drop significantly, stabilizing around 33.61%–33.90%.

These results indicate that a lower regularization coefficient is optimal, as stronger regularization negatively impacts model performance.

---

## Results

### Binary Perceptron (20 Iterations)

- **Training Accuracies:**
  - *Class 1 vs. Class 2:* 99.38%
  - *Class 2 vs. Class 3:* 90.68%  
  - *Class 1 vs. Class 3:* 94.38%

- **Test Accuracies:**
  - *Class 1 vs. Class 2:* 100%
  - *Class 2 vs. Class 3:* 92.50%
  - *Class 1 vs. Class 3:* 89.74%

> _Note:_ The binary classification between class 1 and class 3 performed the worst, suggesting these two classes are difficult to differentiate.

### One-vs-Rest Classification

- **Overall Train Accuracy:** 85.06%
- **Overall Test Accuracy:** 79.66%

### L2 Regularization

- Increasing the regularization coefficient beyond 0.01 resulted in a significant drop in accuracy, with coefficients of 0.1 and higher yielding similar, lower accuracy values (~33%).
- The optimal coefficient was found to be **0.01**.

---

## Observations and Conclusion

- **Perceptron Performance:**  
  The binary perceptron achieved very high accuracy for some class pairs (even reaching 100% on test data for class 1 vs. class 2), but struggled with others (notably class 2 vs. class 3).

- **Multi-Class Classification:**  
  The one-vs-rest strategy effectively converts a multi-class problem into multiple binary classification tasks. However, the overall performance—as reflected by the combined training and testing accuracies—is lower than that of the best binary classifiers.

- **Regularization Impact:**  
  L2 regularization can help control weight magnitudes; nevertheless, too strong a regularization penalty (coefficients ≥ 0.1) leads to underfitting and lower accuracies. A coefficient of 0.01 emerged as the best compromise in our experiments.

- **Overall Insights:**  
  The project highlights the strengths and limitations of the perceptron algorithm for both binary and multi-class classification. Careful hyperparameter tuning—including decisions about iteration count, regularization strength, and classifier strategy—is critical for achieving optimal performance.

---

This project is licensed under the MIT License. See the [LICENSE](https://github.com/clareokafor/Perceptron/blob/main/LICENSE.txt) file for more details.

