# Decision Tree CART and Naive Bayes for Speed Dating Analysis

This project applies Machine Learning methods (CART Decision Trees and Naive Bayes) to analyze and predict outcomes from speed dating events, using data collected by Columbia University.

![Machine Learning Models](https://img.shields.io/badge/ML-Decision%20Tree%20%7C%20Naive%20Bayes-blue)
![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Models Implemented](#models-implemented)
- [Installation and Setup](#installation-and-setup)
- [Running the Models](#running-the-models)
- [Results and Interpretation](#results-and-interpretation)
- [Visualizations](#visualizations)
- [Future Work](#future-work)

## 🎯 Project Overview

This study employs Machine Learning techniques to analyze speed dating events data with the following objectives:

- **Predictive Modeling**: Develop models that can predict whether participants would want to see each other again
- **Feature Analysis**: Identify which attributes significantly influence matching decisions
- **Model Comparison**: Evaluate and compare the performance of Decision Tree and Naive Bayes algorithms
- **Pattern Recognition**: Discover patterns in human dating preferences and behaviors

## 📊 Dataset Description

The dataset comes from speed dating events organized among university students. Each event consisted of 4-minute "first encounters" followed by a questionnaire. The dataset includes:

| Feature | Description |
|---------|-------------|
| `id` | Participant's identifier number |
| `partner` | Peer's identifier number |
| `age` | Participant's age |
| `age_o` | Peer's age |
| `goal` | Participant's objective |
| `date` | Frequency with which the participant goes on dates |
| `go_out` | Frequency with which the participant goes out |
| `int_corr` | Correlation of interests |
| `length` | Opinion regarding the duration of the meeting |
| `met` | Whether the participant already knew the peer (binary) |
| `like` | Whether the participant liked the peer (binary) |
| `prob` | Probability that the peer liked the participant |
| `match` | The target class - whether there was a match (binary) |

## 🤖 Models Implemented

### 1. Decision Tree (CART)
- **Algorithm**: Classification and Regression Tree with Entropy criterion
- **Strengths**: 
  - Easy to interpret and visualize
  - Can handle both numerical and categorical data
  - Naturally models non-linear relationships
- **Implementation Details**:
  - Cross-validation for robust evaluation
  - Feature importance analysis
  - Tree visualization for interpretability

### 2. Naive Bayes
- **Algorithm**: Gaussian Naive Bayes
- **Strengths**:
  - Works well with small datasets
  - Fast training and prediction
  - Handles missing values well
- **Implementation Details**:
  - Feature scaling for better performance
  - Cross-validation for robust evaluation
  - Analysis of class-specific feature distributions

## 🔧 Installation and Setup

### Prerequisites
Ensure you have Python 3.6+ installed, then install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz
```

If you're using Anaconda, you can create a new environment:

```bash
conda create -n dating_ml python=3.8 pandas numpy matplotlib seaborn scikit-learn
conda activate dating_ml
conda install -c anaconda graphviz python-graphviz
```

## 🚀 Running the Models

### Decision Tree
```bash
python "Decision Tree.py"
```

### Naive Bayes
```bash
python "Naive Bayes.py"
```

## 📈 Results and Interpretation

Both models provide:
- Prediction accuracy metrics
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Feature importance analysis

Decision Tree additionally provides a visual representation of the decision-making process, which helps understand how features affect the prediction.

### Sample Interpretation
- **Feature Importance**: The most influential features in predicting matches
- **Confusion Matrix**: Analysis of true positives, false positives, etc.
- **Cross-validation**: Ensuring the model's robustness across different data splits

## 📊 Visualizations

The improved implementation generates several visualizations:
- Decision tree diagram
- Feature importance bar charts
- Confusion matrices
- Class distribution plots

These visualizations are automatically saved as PNG files in your project directory.

## 🔮 Future Work

Potential improvements and extensions:
- Implement ensemble methods (Random Forest, Gradient Boosting)
- Feature engineering to extract more insights
- Hyperparameter tuning for model optimization
- Deploy as a simple web application for interactive exploration
