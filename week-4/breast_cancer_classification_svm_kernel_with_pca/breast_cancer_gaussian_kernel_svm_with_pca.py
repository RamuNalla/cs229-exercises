import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import wandb
import time

df = pd.read_csv("cancer_data.csv")

df.isnull().sum()
df.dropna(axis=1, inplace=True)                           # Data Cleanup from jupyter notebook file

features = ['radius_mean', 'texture_mean']                # Selecting only two important features to make the 2D plot
target = 'diagnosis'

df = df[features + [target]]                              # Modifying the dataset to include features and target

df[target] = df[target].map({'M': 1, 'B': 0})             # Modifying the class labels as 1 and 0 for malignant and benign respectively

X = df[features]                                          
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature Scaling (Avoiding Data leakage)
scaler = StandardScaler()                               # Features are stabdardized so that they have 0 mean and 1 standard deviation

X_train_scaled = scaler.fit_transform(X_train)          # These are numpy arrays after standard scaler
X_test_scaled = scaler.transform(X_test)                # We don't want the model to have any info about the test data during training. All preprocessing steps should fit only on training data and then applied to the test data

# Training the linear SVM model

svm_model = SVC(kernel='linear', C=100)    
# kerner = linear --> the model will try to find linear decision boundary
# c = 1000 --> regularization parameter (soft margin parameter), it controls trade-off between maximizing the margin and minimizing the classification error
# Large value of C means the model will try to classify all training examples correctly (it might result in smaller margib, overfitting risk)
# Smaller C means model will allow some misclassifications to achieve a wider margin

start_time_linear = time.time()
svm_model.fit(X_train_scaled, y_train)                 # Training the model

end_time_linear = time.time()
train_time_linear = end_time_linear - start_time_linear
print(f"Linear SVM training time: {train_time_linear:.4f} seconds")

y_pred = svm_model.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred)
f1_score_linear = f1_score(y_test, y_pred)

print(f"Linear SVM Test Accuracy: {accuracy_linear * 100:.2f}%")
print(f"Linear SVM Test F1 Score: {f1_score_linear:.4f}")


#Training RBF SVM Model on Original data
svm_model_rbf_no_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) # Using default C and gamma='scale'

start_time_rbf_no_pca = time.time()
svm_model_rbf_no_pca.fit(X_train_scaled, y_train)
end_time_rbf_no_pca = time.time()
train_time_rbf_no_pca = end_time_rbf_no_pca - start_time_rbf_no_pca
print(f"RBF SVM (No PCA) training time: {train_time_rbf_no_pca:.4f} seconds")

y_pred_rbf_nopca = svm_model_rbf_no_pca.predict(X_test_scaled)
accuracy_rbf_nopca = accuracy_score(y_test, y_pred_rbf_nopca)
f1_rbf_nopca = f1_score(y_test, y_pred_rbf_nopca) # Calculate F1 score

print(f"RBF SVM (No PCA) Test Accuracy: {accuracy_rbf_nopca * 100:.2f}%")
print(f"RBF SVM (No PCA) Test F1 Score: {f1_rbf_nopca:.4f}")



# Applying PCA

pca = PCA(n_components=0.95, random_state=42)         # Keep the minimum number of components that meet the 95% variance capture threshold
pca.fit(X_train_scaled)

n_components_pca = pca.n_components_
print(f"PCA selected {n_components_pca} components to retain 95% variance.")

cumulative_explained_variance_pca = np.sum(pca.explained_variance_ratio_)
print(f"Cumulative explained variance with {n_components_pca} components: {cumulative_explained_variance_pca:.4f}")

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Train RBF SVM on PCAs

svm_model_rbf_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) # Using default C and gamma='scale'

start_time_rbf_pca = time.time()
svm_model_rbf_pca.fit(X_train_pca, y_train)
end_time_rbf_pca = time.time()
train_time_rbf_pca = end_time_rbf_pca - start_time_rbf_pca
print(f"RBF SVM (PCA data) training time: {train_time_rbf_pca:.4f} seconds")


y_pred_rbf_pca = svm_model_rbf_pca.predict(X_test_pca)
accuracy_rbf_pca = accuracy_score(y_test, y_pred_rbf_pca)
f1_rbf_pca = f1_score(y_test, y_pred_rbf_pca) # Calculate F1 score

print(f"RBF SVM (PCA data) Test Accuracy: {accuracy_rbf_pca * 100:.2f}%")
print(f"RBF SVM (PCA data) Test F1 Score: {f1_rbf_pca:.4f}")




wandb.init(project="cancer-pca-svm-comparison",
           config={
               "n_components_pca_retained": n_components_pca,
               "pca_variance_threshold": 0.95,
               "linear_svm_C": 100,
               "rbf_svm_C": 1.0,
               "rbf_svm_gamma": 'scale'
           })

# Log metrics
wandb.log({
    "cumulative_explained_variance_pca": cumulative_explained_variance_pca,
    "linear_svm_accuracy": accuracy_linear,
    "linear_svm_f1_score": f1_score_linear,
    "linear_svm_training_time": train_time_linear,
    "rbf_svm_nopca_accuracy": accuracy_rbf_nopca, # Log RBF No PCA
    "rbf_svm_nopca_f1_score": f1_rbf_nopca,       # Log RBF No PCA
    "rbf_svm_nopca_training_time": train_time_rbf_no_pca, # Log RBF No PCA
    "pca_rbf_svm_accuracy": accuracy_rbf_pca,
    "pca_rbf_svm_f1_score": f1_rbf_pca,
    "pca_rbf_svm_training_time": train_time_rbf_pca
})

print("Logged results to WandB.")

# Finish the WandB run
wandb.finish()





