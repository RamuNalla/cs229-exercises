import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import wandb

X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)                # Import make_moons dataset with 500 samples
print(f"\nGenerated make_moons data: X shape={X.shape}, y shape={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)                                   # Scaling both X_train and X_tets
X_test_scaled = scaler.transform(X_test)   


## function to plot decision boundary

def plot_svm_boundary(svm_model, X, y, title = "SVM Decision Boundary and Margins", filename = None):

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

    if filename:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()                                    # Closes the plot, while creating multiple plots, if we don't close them, matplotlib keeps them open in memory                    

    plt.show()


results = {}        # Dictionary to store the accuracy results


# Objective is to test multiple Cs with fixed gamma (best gamma=1 found in the previous experiment). 

#-------------------------------RBF KERNEL------------------------------------------------------------------------------------------------------------------------------

WANDb_PROJECT = "svm-make-moons-rbf-tuning" 

print("\n--- Testing RBF Kernel ---")

rbf_params = [(1, 0.01), (1, 1.0), (1, 10), (1, 100), (1, 1000), (1, 10000)]   #Parameters with gamma and C

best_rbf_acc = 0
best_rbf_model = None
best_rbf_params = None

for gamma, C in rbf_params:

    config_rbf = {
        "kernel": "rbf",
        "gamma": gamma,
        "C": C
    }

    run_name_rbf = f"rbf-g{gamma}-C{C}"

    wandb.init(project = WANDb_PROJECT, config=config_rbf, name = run_name_rbf, reinit = True)

    rbf_svm = SVC(kernel='rbf', gamma = gamma, C=C, random_state=42)          # In RBF kernel, the parameters are gamma and C, gamma is a measure of how far the influence of a single training example reacghes
    # High gamma: close points influence each other --> tight decision boundary, risk of overfitting
    # Low gamma: points influence more broadly --> smooth decision boundary, risk of underfitting
    # In Sklearn 'Scale' and 'auto' are default values which are dependent on data
    
    rbf_svm.fit(X_train_scaled, y_train)

    y_pred_rbf = rbf_svm.predict(X_test_scaled)
    
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    f1_score_rbf = f1_score(y_test, y_pred_rbf)
    
    results[f'RBF (gamma={gamma}, C={C})'] = accuracy_rbf
    print(f"RBF SVM (gamma={gamma}, C={C}) Accuracy: {accuracy_rbf:.4f}")
    
    if accuracy_rbf > best_rbf_acc:
        best_rbf_acc = accuracy_rbf
        best_rbf_model = rbf_svm
        best_rbf_params = (gamma, C)


    wandb.log({"test_accuracy": accuracy_rbf})
    wandb.log({"test_f1_score": f1_score_rbf})

    plot_filename_rbf = f"{run_name_rbf}.png"
    plot_svm_boundary(rbf_svm, X, y, f"RBF SVM (gamma={gamma}, C={C}, Acc: {accuracy_rbf:.4f})", filename=plot_filename_rbf)

    wandb.log({"decision_boundary": wandb.Image(plot_filename_rbf)})

    wandb.finish()

print(f"\nBest RBF Model: gamma={best_rbf_params[0]}, C={best_rbf_params[1]}, Accuracy={best_rbf_acc:.4f}")

if best_rbf_model:
    plot_svm_boundary(best_rbf_model, X, y, f'Best RBF SVM (gamma={best_rbf_params[0]}, C={best_rbf_params[1]}, Acc: {best_rbf_acc:.4f})')




#------------------------Comparison of different kernels----------------------

print("\n---Comparison of SVM Kernels")

best_kernel = max(results, key=results.get)        # Results is a dictionary, max for a dictionary by default find the maximum KEY (alphabetically), therefore to tell what maximum to find key argument is used

for kernel, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):   # lambda item: item[1] --> lambda function (short function in a single line). take a tuple item and return item[1]
# results.items() returns a view of the key-value pairs of a dictionary
    print(f"  {kernel}: {acc:.4f}")

print(f"\nBest performing kernel configuration on the test set: {best_kernel} (Accuracy: {results[best_kernel]:.4f})")



