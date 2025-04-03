# E-Commerce Product Tagging with CNNs

## Project Instructions

This project focuses on automating product tagging for an e-commerce store using Convolutional Neural Networks (CNNs).

### Steps to Follow:

1. **Training the Model**:
   - Train the CNN model while keeping the number of epochs to 1 or 2 to minimize runtime.
   - After training, store the predictions on the test set in a list named `predictions`.

2. **Evaluating the Model**:
   - Calculate the following metrics based on the predictions:
     - **Accuracy**: Store the overall accuracy in a variable named `accuracy`.
     - **Per-Class Precision and Recall**: 
       - Calculate precision and recall for each class.
       - Store these metrics in lists named `precision` and `recall`, respectively, with appropriate lengths corresponding to the number of classes.

### Notes:
- Ensure the predictions and metrics are stored in the specified variables (`predictions`, `accuracy`, `precision`, and `recall`).
- Use appropriate methods and libraries to compute the metrics efficiently.

### Objective:
The goal is to create a robust product tagging system that can classify e-commerce products accurately and provide detailed performance metrics for each class.