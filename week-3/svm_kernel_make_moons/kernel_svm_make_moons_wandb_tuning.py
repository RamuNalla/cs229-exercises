import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import wandb

WANDb_PROJECT = "svm-make-moons-tuning" 



X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)                # Import make_moons dataset with 500 samples
print(f"\nGenerated make_moons data: X shape={X.shape}, y shape={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)                                   # Scaling both X_train and X_tets
X_test_scaled = scaler.transform(X_test)   


## function to plot decision boundary and saving the plot with a filename

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





# ------------------------------------ Linear SVM -------------------------------------------------------------------------------------------------------------------------------------

print("\n--- Testing Linear Kernel ---")

C_linear = 1.0

config_linear = {
    "kernel": "linear",
    "C": C_linear
}

run_name_linear = f"linear-C_{C_linear}"

wandb.init(project=WANDb_PROJECT, config=config_linear, name=run_name_linear, reinit=True)    # reinit = True --> controls we can reinitialize wandb in the same python sessiom

linear_svm = SVC(kernel="linear", C = C_linear, random_state=42)       # random_state ensure model trains same way every time you run it as it involves randomness wih the underlying optimization
linear_svm.fit(X_train_scaled, y_train)

y_pred_linear = linear_svm.predict(X_test_scaled)

accuracy_linear = accuracy_score(y_test, y_pred_linear)
results['Linear (C=1)'] = accuracy_linear                        # Dictionary entry for C=1 linear SVM
print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")




# Log metriccs to wandb

wandb.log({"test_accuracy": accuracy_linear})

plot_filename_linear = f"{run_name_linear}.png"
plot_svm_boundary(linear_svm, X, y, f'Linear SVM (Accuracy: {accuracy_linear:.4f})', filename=plot_filename_linear)

wandb.log({"decision_boundary": wandb.Image(plot_filename_linear)})

wandb.finish()  # End the WandB run

#Since the plot closes in the wandb call, calling it again without filename to show while the code is executing
plot_svm_boundary(linear_svm, X, y, f'Linear SVM (Accuracy: {accuracy_linear:.4f})')


#-------------------------------POLYNOMINAL KERNEL------------------------------------------------------------------------------------------------------------------------------

print("\n--- Testing Polynomial Kernel ---")

poly_params = [(3, 1.0), (3, 10.0), (4, 1.0)]                   # Parameters with degree and C

best_poly_accuracy = 0                                          # Out of three which is the best poly kernel, initializing the data
best_poly_model = None
best_poly_params = None


for degree, C in poly_params:

    #wandb hyperparameters
    config_poly = {
        "kernel": "poly",
        "degree": degree,
        "C": C
    }
    
    run_name_poly = f"poly-d{degree}-C{C}"

    wandb.init(project = WANDb_PROJECT, config=config_poly, name = run_name_poly, reinit = True)
    
    
    poly_svm = SVC(kernel="poly", degree=degree, C=C, random_state=42)      # In polynomial kernel, degree is the degree of the polynomial and C is the regularization parameter (trade-off between maximizing the margin and minimizing classification error)
    # Smaller C → larger margin, more misclassifications allowed.
    # Larger C → stricter margin, fewer violations allowed
    
    poly_svm.fit(X_train_scaled, y_train)

    y_pred_poly = poly_svm.predict(X_test_scaled)

    accuracy_poly = accuracy_score(y_test, y_pred_poly)

    results[f'Poly (degree = {degree}, C = {C})'] = accuracy_poly
    print(f"Polynomial SVM (degree={degree}, C={C}) Accuracy: {accuracy_poly:.4f}")

    wandb.log({"test_accuracy": accuracy_poly})

    plot_filename_poly = f"{run_name_poly}.png"
    plot_svm_boundary(poly_svm, X, y, f"Poly SVM (d={degree}, C={C}, Acc: {accuracy_poly:.4f})", filename=plot_filename_poly)

    wandb.log({"decision_boundary": wandb.Image(plot_filename_poly)})

    wandb.finish()


    if accuracy_poly > best_poly_accuracy:
        best_poly_accuracy = accuracy_poly
        best_poly_model = poly_svm
        best_poly_params = (degree, C)

print(f"\nBest Polynomial Model: degree={best_poly_params[0]}, C={best_poly_params[1]}, Accuracy={best_poly_accuracy:.4f}")

# Plotting only the best model with polynomian kernel
if best_poly_model:
    plot_svm_boundary(best_poly_model, X, y, f'Best Poly SVM (d={best_poly_params[0]}, C={best_poly_params[1]}, Acc: {best_poly_accuracy:.4f})')






#-------------------------------RBF KERNEL------------------------------------------------------------------------------------------------------------------------------

print("\n--- Testing RBF Kernel ---")

rbf_params = [('scale', 1.0), ('scale', 10.0), (0.1, 1.0), (1, 1.0), (1, 10.0)]   #Parameters with gamma and C

best_rbf_acc = 0
best_rbf_model = None
best_rbf_params = None

for gamma, C in rbf_params:


    config_rbf = {
        "kernel": "rbf",
        "gamma": gamma,
        "C": C
    }
    
    #Some of gammas are 'scale', therefore handling it
    if gamma == 'scale':
        gamma_str = gamma
    else:
        gamma_str = f"{gamma:.1f}"
    
       
    run_name_rbf = f"rbf-g{gamma_str}-C{C}"

    wandb.init(project = WANDb_PROJECT, config=config_rbf, name = run_name_rbf, reinit = True)
    



    rbf_svm = SVC(kernel='rbf', gamma = gamma, C=C, random_state=42)          # In RBF kernel, the parameters are gamma and C, gamma is a measure of how far the influence of a single training example reacghes
    # High gamma: close points influence each other --> tight decision boundary, risk of overfitting
    # Low gamma: points influence more broadly --> smooth decision boundary, risk of underfitting
    # In Sklearn 'Scale' and 'auto' are default values which are dependent on data
    
    rbf_svm.fit(X_train_scaled, y_train)

    y_pred_rbf = rbf_svm.predict(X_test_scaled)
    
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    
    results[f'RBF (gamma={gamma}, C={C})'] = accuracy_rbf
    print(f"RBF SVM (gamma={gamma}, C={C}) Accuracy: {accuracy_rbf:.4f}")
    


    wandb.log({"test_accuracy": accuracy_rbf})

    plot_filename_rbf = f"{run_name_rbf}.png"
    plot_svm_boundary(rbf_svm, X, y, f"RBF SVM (gamma={gamma_str}, C={C}, Acc: {accuracy_rbf:.4f})", filename=plot_filename_rbf)

    wandb.log({"decision_boundary": wandb.Image(plot_filename_rbf)})

    wandb.finish()


    if accuracy_rbf > best_rbf_acc:
        best_rbf_acc = accuracy_rbf
        best_rbf_model = rbf_svm
        best_rbf_params = (gamma, C)

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



