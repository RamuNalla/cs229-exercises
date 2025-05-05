import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import wandb

WANDB_PROJECT = "svm-moons-analysis"

X, y = make_moons(n_samples=1000, noise=0.25, random_state=41)                # Import make_moons dataset with 500 samples
print(f"\nGenerated make_moons data: X shape={X.shape}, y shape={y.shape}")


# Data split into Train, Validation and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Typically again split into train and validation is not done for CV methods, as CV itself creates a subset for validation

print(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

# Data normalization

scaler = StandardScaler()
scaler.fit(X_train)        

X_train_scaled = scaler.transform(X_train)                                   # Scaling both X_train and X_tets using X_train fit model
X_test_scaled = scaler.transform(X_test)  

print(f"Scaled data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")

## function to plot decision boundary

def plot_svm_boundary(svm_model, X_train, y_train, X_test, y_test, scaler, title = "SVM Decision Boundary and Margins", filename = None):

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
        plt.savefig(filename) # Save the plot
        print(f"Saved plot to {filename}")
        plt.close() # Close the plot to avoid displaying it inline if saving
    else:
        plt.show()


results = {}        # Dictionary to store the accuracy results


#-------------------------------RBF FIXED SVM ------------------------------------------------------------------------------------------------------------------------------

print("\n--- Evaluating Fixed SVM with cross val score---")


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


# Log the GridsearchCV results to wandb

run_name_grid = f"gridsearch-{best_svm_model.kernel}-C{best_svm_model.C}-g{best_svm_model.gamma}"
wandb.init(project=WANDB_PROJECT, config=grid.best_params_, name = run_name_grid, reinit=True)

wandb.log({
    "best_cv_score": grid.best_score_,

    "best_C": best_svm_model.C,
    "best_gamma": best_svm_model.gamma if isinstance(best_svm_model.gamma, (int, float)) else str(best_svm_model.gamma),
    "best_kernel": best_svm_model.kernel
})






print("\n--- Accuracy with the Test data using best SVM model---")

y_test_pred = best_svm_model.predict(X_test_scaled)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy of Best Model: {accuracy_test:.4f}")

wandb.log({"test_accuracy": accuracy_test})





plot_filename_boundary = f"{run_name_grid}_boundary.png"
plot_svm_boundary(best_svm_model, X_train, y_train, X_test, y_test, scaler, title=f"Best SVM (GridSearchCV): {grid.best_params_}\nTest Acc: {accuracy_test:.4f}", filename=plot_filename_boundary)
wandb.log({"decision_boundary": wandb.Image(plot_filename_boundary)})
wandb.finish() # Finish the GridSearchCV run



### ------------GENERATING LEARNING CURVE FOR THE BEST MODEL------------------------------

# To see how the model's performance changes with increase in training data size

# input the model and training data
# train_sizes = how many different train sizes you want to plot the learning curve on. Here 0.1 to 1 with 10 different sizes: 10%, 20%,...,100% of the data
# cv = 5, for each training data set the CV is done on 5 sets --> and calculated training score and validations score
# scoring = accuracy --> accuracy as the measure
# n_jobs = -1 --> use all available CPU cores

# Output:
# train_sizes_abs: an array of actual number of training examples used in each step. Here [70, 140, 210,...,700]
# train scores: 2d array of accuracy scores on the training set for each fold and each training set (10 rows and 5 columns). Note CV=5 means, everytime 4 folds is used to train the model and calculate the training error
# Validation scores: same but of validations score
# The shape of 2D array is n_train_sizes x n_folds --> (10, 5) matrix

print(f"Scaled data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")

train_sizes_abs, train_scores, validation_scores = learning_curve(
    estimator=best_svm_model,
    X = X_train_scaled,
    y = y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv = 5,
    scoring = 'accuracy',
    n_jobs=-1,
    random_state=42
)

# The above piece of code is running an experiment that trains the model on increasing subsets of training data and records how ell it performs each time


#Calculate the mean and std for training and validation scores
train_scores_mean = np.mean(train_scores, axis = 1)  #row wise mean --> mean for each training size
train_scores_std = np.std(train_scores, axis=1)

validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.figure(figsize = (10,6))
plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label = "Training score")
plt.plot(train_sizes_abs, validation_scores_mean, 'o-', color='g', label = "Validation score")


# Add shaded regions for standard deviation
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes_abs, validation_scores_mean - validation_scores_std,
                 validation_scores_mean + validation_scores_std, alpha=0.1, color="g")


plt.title("Learning Curve for Best SVM Model")
plt.xlabel("Number of Training Examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.8, 1.01) # Adjust ylim for better visibility if needed

learning_curve_filename = f"{run_name_grid}_learning_curve.png"
plt.savefig(learning_curve_filename)
print(f"Saved learning curve plot to {learning_curve_filename}")



# Log the learning plot to wandb

run_name_lc = f"learningcurve-{best_svm_model.kernel}-C{best_svm_model.C}-g{best_svm_model.gamma}"
wandb.init(project=WANDB_PROJECT, config=grid.best_params_, name=run_name_lc, reinit=True)
wandb.log({"learning_curve_plot": wandb.Image(learning_curve_filename)})
wandb.finish()

plt.show() 

# Note the learning curev is till 560 samples only because in k-fold, with CV=5, only 4 subsets is used for training, therefore 700*0.8 = 560 samples


#Analysis of Learning Curve ---


print("\n--- Learning Curve Analysis ---")
if validation_scores_mean[-1] < 0.9: # Example threshold, adjust as needed
     print("Low Convergence Score: The cross-validation score appears to converge to a relatively low value.")
     print("Suggestion: This might indicate high bias (underfitting). Consider a more complex model (e.g., different kernel, more features if applicable), or adjusting hyperparameters like C/gamma.")
elif (train_scores_mean[-1] - validation_scores_mean[-1]) > 0.05: # Example threshold for gap
     print("Large Gap: There's a noticeable gap between the training score and the cross-validation score.")
     print("Suggestion: This might indicate high variance (overfitting). Consider getting more training data, increasing regularization (e.g., decreasing C for SVM), or using a simpler model.")
else:
     print("Good Convergence: The training and cross-validation scores seem to converge to a high value with a small gap.")
     print("Suggestion: The model seems well-suited for the data size and complexity. Further improvements might require more data or more advanced feature engineering.")

print(f"(Final Training Score: {train_scores_mean[-1]:.4f}, Final CV Score: {validation_scores_mean[-1]:.4f})")



