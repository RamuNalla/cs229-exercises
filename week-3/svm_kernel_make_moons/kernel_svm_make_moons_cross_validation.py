import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=0.25, random_state=41)                # Import make_moons dataset with 500 samples
print(f"\nGenerated make_moons data: X shape={X.shape}, y shape={y.shape}")


# Data split into Train, Validation and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Typically again split into train and validation is not done for CV methods, as CV itself creates a subset for validation

print(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

# Data normalization

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)                                   # Scaling both X_train and X_tets
X_test_scaled = scaler.transform(X_test)  



## function to plot decision boundary

def plot_svm_boundary(svm_model, X_train, y_train, X_test, y_test, title = "SVM Decision Boundary and Margins"):

    plt.figure(figsize=(8, 6))
    
    h = 0.02    # Step size in the meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xy = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
  
    Z = svm_model.predict(xy).reshape(xx.shape)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', label = 'Train data')    #c=y means set the color of each point based on label 'y', s=50 means size of each point in the scatter plot
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker = 'x', alpha=0.6, label = 'Test data')    #c=y means set the color of each point based on label 'y', s=50 means size of each point in the scatter plot
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)                  
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


results = {}        # Dictionary to store the accuracy results


#-------------------------------RBF FIXED SVM ------------------------------------------------------------------------------------------------------------------------------

print("\n--- Evaluating Fixed SVM on RBG kernel---")


rbf_fixed_svm = SVC(kernel = "rbf", gamma = "scale", C = 1.0, random_state=42)

cv_scores = cross_val_score(rbf_fixed_svm, X_train_scaled, y_train, cv=5)

print("\n--- Fixed 5-Fold Cross-Validation Scores ---")
print(f"Cross-validation scores (5 folds) for fixed SVM (RBF, C=1, gamma='scale'): {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")



#------------------------HYPERPARAMETER TUNING USING GRIDSEARCHCV---------------------------------

print("\n--- Hyperparameter tuning with GridsearchCV---")

params_grid = {
    "kernel" : ["linear", "rbf"],
    "gamma" : ["scale", "auto", 0.1, 1.0, 10], 
    "C" : [0,1, 1, 10, 100]
}

# GridsearchCV finds best hyperparameters combination using cross-validation
# Inputs - Model, Grid of parameters, Scoring metric, CV strategy
# best_params_: Best hyperparameter copmbination
# best_score_: best cross validation score
# best_estimator_:best model that is trained with the best settings

grid = GridSearchCV(SVC(random_state=42), param_grid = params_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Run the model on the dataset
grid.fit(X_train_scaled, y_train)

print("\n--- GridSearchCV Results ---")
print(f"Best Parameters found: {grid.best_params_}")
print(f"Best Cross-Validation Score (Mean Accuracy): {grid.best_score_:.4f}")


best_svm_model = grid.best_estimator_    # The best model trained on best parameters

print("\n--- Accuracy with the Test data using best SVM model---")

y_test_pred = best_svm_model.predict(X_test_scaled)

accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Test Set Accuracy of Best Model: {accuracy_test:.4f}")

plot_svm_boundary(best_svm_model, X_train, y_train, X_test, y_test, title=f"Best SVM (GridSearchCV): {grid.best_params_}\nTest Acc: {accuracy_test:.4f}")






