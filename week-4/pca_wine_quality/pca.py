import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import wandb




#-----------------STEP-1: DATA LOADING AND EDA--------------------------------------------

df = pd.read_csv("./wine_quality_red.csv")

# df.info()                                  # Gives quick overview of the data
# print(df.describe())                       # Gives statistical information for all numerical columns
# print(df.isnull().sum())                   # checks for missing values in each column
# print(df.head())

# df.hist(figsize=(20, 10), bins=20)           # Plotting 12 histograms to see the data variation
# plt.tight_layout()
# plt.show()

#-----------------STEP-2: DATA SCALING AND TRAIN/TEST SPLIT--------------------------------------------

numerical_cols = df.select_dtypes(include=np.number).columns.to_list()     # The numerical columns in the dataset


# Separate data in X and y: Remove quality from df and make it X, call the column 'quality' as y

X = df[numerical_cols].drop('quality', axis = 1)  
y = df['quality'] 

print(f"Shape of Features (X): {X.shape}")
print(f"Shape of Target (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   #Statify helps maintian the y distribution preserved in both training and test splits

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#-----------------STEP-3: CALCULATE COVARIANCE MATRIX AND ITS E.VALUES AND E.VECTORS--------------------------------------------

covariance_matrix = np.cov(X_train_scaled, rowvar = False)     # rowvar = True means each row is treated as variable

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

print("\nEigenvalues of the Covariance Matrix:")
print(eigenvalues)


#-----------------STEP-4: PCA IMPLEMENTATION FROM SCRATCH--------------------------------------------


# Sorting the eigen values in descending order and sort corresponding eigen vectors
sorted_indices = np.argsort(eigenvalues)[::-1]                 # np.argsort(eigenvalues) sorts the indices in ascending order, adding [::-1] makes it reverse
sorted_eigen_values = eigenvalues[sorted_indices]
sorted_eigen_vectors = eigenvectors[:, sorted_indices]          
print("\nSorted Eigenvalues (Descending):")
print(sorted_eigen_values)

k = 2                                                          # Number of components to keep (better for visualization)

top_k_eigen_vectors = sorted_eigen_vectors[:, :k]              # selecting top k eigen vectors
print(f"\nSelected Top {k} Eigenvectors (Projection Matrix shape): {top_k_eigen_vectors.shape}")


# Project the scaled training data into the principal components. Now the data will be (n_training_samples, k)
X_projected = X_train_scaled @ top_k_eigen_vectors  

print(f"\nProjected Data (X_projected) shape: {X_projected.shape}")


# Calculation of variance explained by each component

total_variance = np.sum(sorted_eigen_values)
explained_variance_ratio = sorted_eigen_values / total_variance                 # It is an array of (11,)

cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nExplained Variance Ratio (from scratch):")
print(explained_variance_ratio)

print("\nCumulative Explained Variance (from scratch):")
print(cumulative_variance)



#-----------------STEP-5: PCA IMPLEMENTATION WITH SKLEARN--------------------------------------------

sklearn_pca = PCA(n_components=k, random_state=42)

sklearn_pca.fit(X_train_scaled)

X_projected_sklearn = sklearn_pca.fit_transform(X_train_scaled)          # Project X into the k-components

print(f"\nProjected Data (X_projected_sklearn) shape: {X_projected_sklearn.shape}")

print("\nExplained Variance Ratio (Sklearn):")
print(sklearn_pca.explained_variance_ratio_)

print("\nCumulative Explained Variance (Sklearn):")
print(np.cumsum(sklearn_pca.explained_variance_ratio_))


#-----------------STEP-6: COMPARISON BETWEEN SCRATCH IMPLEMENTATION and SKLEARN--------------------------------------------


print("\nComparison of Projected Data (Scratch vs Sklearn) with a correlation matrix:")

correlation_matrix = np.corrcoef(X_projected.T, X_projected_sklearn.T)   # Pearson correlation coefficient between all pairs of rows, it should be close to identity or negative identity
print("Correlation matrix between Scratch and Sklearn projected:\n", correlation_matrix)



#-----------------STEP-7: WANDB INITIALIZATION and LOGGING--------------------------------------------

wandb.init(project="wine-quality-pca", name=f"pca_k={k}")
wandb.config.k = k

wandb.log({
    "cumulative_explained_variance_scratch": cumulative_variance[k-1],
    "cumulative_explained_variance_sklearn": np.cumsum(sklearn_pca.explained_variance_ratio_)[k-1]
})

#-----------------STEP-8: VISUALIZE PROJECTED DATA--------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(X_projected[:, 0], X_projected[:, 1],c=y_train, cmap='viridis', s=50, alpha=0.7) # Use y_train for coloring

plt.title('PCA Projected Data (PC1 vs PC2) colored by Wine Quality')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Wine Quality') # Add a color bar for the quality levels
plt.grid(True, linestyle='--', alpha=0.6)

wandb.log({"PCA_Projection_Plot": wandb.Image(plt)})
wandb.finish()
plt.show()

