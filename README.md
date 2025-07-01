# K-Nearest Neighbors (KNN) Classification

## Overview

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm for a multi-class classification problem using the Iris dataset. The notebook includes data preprocessing, model training, evaluation, and visualization of decision boundaries.

---

## Objective

- Understand and implement KNN for classification tasks.
- Learn the importance of feature normalization.
- Evaluate the model using appropriate metrics.
- Explore the impact of varying the number of neighbors (`K`) on performance.
- Visualize decision boundaries to interpret model behavior.

---

## Dataset

The Iris dataset contains measurements of iris flowers from three different species:

- **Features:**
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
- **Target:**
  - Species (Iris-setosa, Iris-versicolor, Iris-virginica)

No missing values are present in the dataset. The target column is encoded numerically for model training.

---

## Implementation Steps

1. **Data Loading and Exploration**
   - Load the dataset and inspect its structure.
   - Check for missing values.

2. **Preprocessing**
   - Encode the target column (Species) using Label Encoding.
   - Normalize feature columns using StandardScaler to ensure all features contribute equally.

3. **Train-Test Split**
   - Split the data into training and testing sets with an 80%-20% ratio.

4. **Model Training**
   - Use `KNeighborsClassifier` from Scikit-learn.
   - Start with `K=5` and train the model on the training data.

5. **Model Evaluation**
   - Evaluate the model using accuracy, confusion matrix, and classification report.
   - Experiment with different values of `K` (from 1 to 20) to analyze their effect on accuracy.
   - Visualize accuracy trends with respect to `K`.

6. **Decision Boundary Visualization**
   - For simplicity, visualize decision boundaries using only PetalLengthCm and PetalWidthCm features.
   - Display how the model classifies different regions in feature space.

---

## Results

- **Accuracy for K=5:** 0.93
- **Confusion Matrix:**
  - All Iris-setosa samples correctly classified.
  - Minor misclassifications between Iris-versicolor and Iris-virginica.
- **Classification Report:**
  - High precision, recall, and F1-scores across all classes.

The model achieves strong overall performance, demonstrating KNN's effectiveness on this dataset.

---

## Key Observations

- Feature normalization is essential in KNN as it relies on distance metrics.
- Choice of `K` significantly affects the bias-variance trade-off.
- Visualization of decision boundaries helps understand class separability and model behavior.
- KNN performs well when classes are well-separated but can struggle when classes overlap.



---

## Conclusion

This implementation showcases how KNN can effectively classify iris species with high accuracy when appropriate preprocessing and parameter tuning are applied. The experiment also highlights the importance of normalization and careful choice of `K` to achieve optimal results.

---

## Repository Contents

- `knn_iris_classification.ipynb`: Jupyter Notebook containing complete code, visualizations, and analysis.
- `Iris.csv`: Dataset file used for this analysis.
- `README.md`: Documentation describing project details, setup instructions, and key results.

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/neighbors.html
- UCI Machine Learning Repository: Iris Dataset

---

