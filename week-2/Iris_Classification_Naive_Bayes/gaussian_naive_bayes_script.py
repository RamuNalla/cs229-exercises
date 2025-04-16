import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

from gaussian_naive_bayes_class import GaussianNaiveBayes     # Importing the class from the other file

try:
    from sklearn.naive_bayes import GaussianNB as SklearnGNB
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -------------------------------- GAUSSIAN NAIVE BAYES FROM SCRATCH -------------------------------------

if __name__ == "__main__":                   # This will ensure the below code is run only when the script is executed directly not when this python file is imported to some other code

    print("Running Gaussian Naive Bayes Implementation...")

    print("\n1. Loading Iris dataset...")
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    print(f"   Dataset shape: Features X={X.shape}, Labels y={y.shape}")
    print(f"   Target names: {iris.target_names}") # Names corresponding to labels 0, 1, 2
    print(f"   Unique labels found in y: {np.unique(y)}")

    print("\n2. Splitting data into training and test sets (80/20 split)...")

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)        # Stratify= y will ensure class proportions are same in both train and test sets

    print(f"   Training set size: {X_train.shape[0]} samples")
    print(f"   Test set size: {X_test.shape[0]} samples")

    gnb_custom = GaussianNaiveBayes()       # creating an instance of the GaussianNaiveBayes class

    gnb_custom.fit(X_train, y_train)

    y_pred_custom = gnb_custom.predict(X_test)

    accuracy_custom = sklearn.metrics.accuracy_score(y_test, y_pred_custom)
    print(f"--> Accuracy of custom GNB model: {accuracy_custom:.4f}")


# -------------------------------- GAUSSIAN NAIVE BAYES SKLEARN IMPLEMENTATION -------------------------------------

    if SKLEARN_AVAILABLE:
        
        gnb_sklearn = SklearnGNB()

        gnb_sklearn.fit(X_train, y_train)

        sklearn_y_pred = gnb_sklearn.predict(X_test)

        sklearn_accuracy = sklearn.metrics.accuracy_score(y_test, sklearn_y_pred)
        print(f"--> Accuracy of scikit-learn GNB model: {sklearn_accuracy:.4f}")
    else:
        print("\n6. Scikit-learn not found, skipping comparison.")












