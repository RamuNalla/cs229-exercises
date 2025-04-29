import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=0.25, random_state=41)                # Import make_moons dataset with 500 samples
print(f"\nGenerated make_moons data: X shape={X.shape}, y shape={y.shape}")


# Data split into Train, Validation and Test sets
X_train_total, X_test, y_train_total, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
X_train, X_val, y_train, y_val = train_test_split(X_train_total, y_train_total, test_size=0.2, random_state=42) 

print(f"Split data: Train={X_train.shape}, Validation = {X_val.shape}, Test={X_test.shape}")

# Data normalization

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)                                   # Scaling both X_train and X_tets
X_val_scaled = scaler.transform(X_val)
X_train_total_scaled = scaler.transform(X_train_total)
X_test_scaled = scaler.transform(X_test)  



## function to plot decision boundary

def plot_svm_boundary(svm_model, X, y, title = "SVM Decision Boundary and Margins"):

    plt.figure(figsize=(8, 6))
    
    h = 0.02    # Step size in the meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xy = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
  
    Z = svm_model.predict(xy).reshape(xx.shape)

    plt.scatter(X_train_total[:, 0], X_train_total[:, 1], c=y_train_total, cmap=plt.cm.coolwarm, edgecolors='k', label = 'Train data')    #c=y means set the color of each point based on label 'y', s=50 means size of each point in the scatter plot
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker = 'x', alpha=0.6, label = 'Test data')    #c=y means set the color of each point based on label 'y', s=50 means size of each point in the scatter plot
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)                  
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


results = {}        # Dictionary to store the accuracy results


#-------------------------------RBF KERNEL------------------------------------------------------------------------------------------------------------------------------

print("\n--- Testing RBF Kernel ---")

rbf_params = [('scale', 1.0), ('scale', 10.0), (0.1, 1.0), (1, 1.0), (1, 10.0)]   #Parameters with gamma and C

best_rbf_acc = 0
best_rbf_model = None
best_rbf_params = None

for gamma, C in rbf_params:

    rbf_svm = SVC(kernel='rbf', gamma = gamma, C=C, random_state=42)          # In RBF kernel, the parameters are gamma and C, gamma is a measure of how far the influence of a single training example reacghes
    # High gamma: close points influence each other --> tight decision boundary, risk of overfitting
    # Low gamma: points influence more broadly --> smooth decision boundary, risk of underfitting
    # In Sklearn 'Scale' and 'auto' are default values which are dependent on data
    
    rbf_svm.fit(X_train_scaled, y_train)

    y_pred_rbf = rbf_svm.predict(X_val_scaled)
    
    accuracy_rbf = accuracy_score(y_val, y_pred_rbf)
    
    results[f'RBF (gamma={gamma}, C={C})'] = accuracy_rbf
    print(f"RBF SVM (gamma={gamma}, C={C}) Accuracy: {accuracy_rbf:.4f}")
    
    if accuracy_rbf > best_rbf_acc:
        best_rbf_acc = accuracy_rbf
        best_rbf_model = rbf_svm
        best_rbf_params = (gamma, C)
        best_gamma = gamma
        best_C = C

print(f"\nBest RBF Model: gamma={best_rbf_params[0]}, C={best_rbf_params[1]}, Accuracy={best_rbf_acc:.4f}")


#------------------------Retraining the model on full training dataset---------------------------------


best_rbf_svm = SVC(kernel='rbf', gamma = best_gamma, C=best_C, random_state=42)          

best_rbf_svm.fit(X_train_total_scaled, y_train_total)


#------------------------Test the full model on held-out Test data---------------------------------


y_pred_test_rbf = best_rbf_svm.predict(X_test_scaled)  
accuracy_test_rbf = accuracy_score(y_test, y_pred_test_rbf)
    
print("\n--- Testing Best Model on test data ---")
print(f"RBF SVM (gamma={best_gamma}, C={best_C}) Accuracy: {accuracy_test_rbf:.4f}")


if best_rbf_model:
    plot_svm_boundary(best_rbf_svm, X, y, f'Best RBF SVM (gamma={best_gamma}, C={best_C}, Acc: {best_rbf_acc:.4f})')

