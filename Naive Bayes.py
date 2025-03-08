import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """Load and preprocess the speed dating dataset."""
    # Column names of the table
    cn = ['id', 'partner', 'age', 'age_o', 'goal', 'date', 'go_out', 'int_corr', 'length', 'met', 'like', 'prob', 'match']
    
    # Load the dataset
    dataset = pd.read_csv("speedDating_trab.csv", header=0, names=cn)
    
    # Pre-processing
    # Remove rows with NA occurrences and unnecessary attributes (id and partner)
    dataset = dataset.dropna().drop(['id', 'partner'], axis=1)
    
    return dataset

def train_naive_bayes_model(X_train, y_train):
    """Train a Gaussian Naive Bayes model."""
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    # Predict the match
    prediction = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, prediction) * 100
    
    # Generate classification report
    report = classification_report(y_test, prediction)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, prediction)
    
    return accuracy, report, cm, prediction

def plot_confusion_matrix(cm):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Match', 'Match'],
                yticklabels=['No Match', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('naive_bayes_confusion_matrix.png')
    plt.show()

def main():
    """Main function to run the Naive Bayes analysis."""
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Display basic dataset information
    print(f"Dataset shape: {dataset.shape}")
    print(f"Class distribution:\n{dataset['match'].value_counts()}")
    
    # Split the dataset into training and testing sets
    X = dataset.drop('match', axis=1)
    y = dataset['match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
    
    # Optional: Feature scaling (Gaussian NB works well with normalized features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    gnb = train_naive_bayes_model(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(GaussianNB(), X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    
    # Evaluate the model
    accuracy, report, cm, prediction = evaluate_model(gnb, X_test_scaled, y_test)
    
    # Display results
    print(f"Gaussian Naive Bayes prediction accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Feature importance (using mean and variance per class)
    feature_names = X.columns
    class_var = []
    for i in range(len(gnb.classes_)):
        class_var.append(gnb.var_[i])
    
    feature_importance = np.abs(np.array(class_var[1]) - np.array(class_var[0]))
    indices = np.argsort(feature_importance)[::-1]
    
    print("\nFeature importance (based on class variance differences):")
    for i, idx in enumerate(indices[:5]):  # Top 5 features
        print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

if __name__ == "__main__":
    main()
