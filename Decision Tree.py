import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz

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

def train_decision_tree(X_train, y_train, max_depth=None):
    """Train a decision tree classifier."""
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=max_depth)
    clf.fit(X_train, y_train)
    return clf

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
    plt.savefig('decision_tree_confusion_matrix.png')
    plt.show()

def visualize_tree(model, feature_names, class_names=None):
    """Visualize the decision tree."""
    # Text representation
    text_representation = tree.export_text(model, feature_names=feature_names)
    print(text_representation)
    
    # Visual representation
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True, 
              rounded=True, 
              fontsize=8)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png')
    plt.show()
    
    # GraphViz export (more detailed)
    dot_data = tree.export_graphviz(model, 
                                   feature_names=feature_names,
                                   class_names=class_names,
                                   filled=True, 
                                   rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_graphviz")
    
    return graph

def main():
    """Main function to run the Decision Tree analysis."""
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Display basic dataset information
    print(f"Dataset shape: {dataset.shape}")
    print(f"Class distribution:\n{dataset['match'].value_counts()}")
    
    # Split features and target
    X = dataset.drop('match', axis=1)
    y = dataset['match']
    
    # Split the dataset into training and testing sets (improved split method)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model (with controlled max_depth to prevent overfitting)
    clf = train_decision_tree(X_train, y_train, max_depth=5)
    
    # Cross-validation
    cv_scores = cross_val_score(DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5), 
                                X, y, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
    
    # Evaluate the model
    accuracy, report, cm, prediction = evaluate_model(clf, X_test, y_test)
    
    # Display results
    print(f"Decision Tree prediction accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Feature importance
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('decision_tree_feature_importance.png')
    plt.show()
    
    # Print top 5 important features
    indices = np.argsort(feature_importance)[::-1]
    print("\nFeature importance:")
    for i, idx in enumerate(indices[:5]):  # Top 5 features
        print(f"{i+1}. {X.columns[idx]}: {feature_importance[idx]:.4f}")
    
    # Visualize the tree
    visualize_tree(clf, feature_names=X.columns, class_names=['No Match', 'Match'])

if __name__ == "__main__":
    main()