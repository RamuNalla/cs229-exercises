import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from matplotlib.patches import Ellipse
from scipy.stats import chi2



#-----------------STEP-1: DATA LOADING AND SCALING--------------------------------------------

df = pd.read_csv("./mall_customers.csv")
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#-----------------STEP-2: FUNCTION TO DRAW ELLIPSES WITH A GIVEN POSITION AND COVARIANCES--------------------------------------------

def draw_ellipse(position, covariance, ax, **kwargs):                         # **kwargs is a python syntax that can accept any number of keyword arguments
    """Draw an ellipse with a given position and covariance"""
    # Convert covariance to principal axes
    if covariance.shape == (2, 2): # full covariance
        U, s, Vt = np.linalg.svd(covariance)                                 # SVD gives U, S, T, U is a 2x2 matrix that gives directions of ellipse axes, S is a diagonal matrix with scales along those axes, VT is not used here
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))                     # Angle the ellipse to be rotated w.r.t X axis
        width, height = 2 * np.sqrt(s)                                       # sqrt gives standard deviations and multiplying by 2 to get full axis lengths
    elif covariance.shape == (2,): # diag covariance
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    elif covariance.shape == (): # spherical covariance
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    else:
        raise ValueError("Invalid covariance shape")

    # Draw the Ellipse
    # We scale the ellipse by a factor related to chi2 for a confidence interval
    # e.g., chi2.ppf(0.95, 2) for ~2-sigma equivalent in 2D Gaussian
    scale_factor = chi2.ppf(0.95, 2) # ~2-sigma equivalent for 2 degrees of freedom (for 95% confidence)
    ellipse = Ellipse(position, width * np.sqrt(scale_factor), height * np.sqrt(scale_factor), angle=angle, **kwargs)
    ax.add_patch(ellipse)




#-----------------STEP-3: SKLEARN GAUSSIAN MIXTURE--------------------------------------------

k = 5                                  # Number of clusters = 4
covariance_type = 'full'               # Each cluster has its own full covariance matrix
# Other options: "diag' - diagonal covariance (no inter-feature correlation)
# 'tied' - all clusters share same full covariance
# 'spherical' - each cluster has a single variance (simplest, like K-Means)

# n_init = 10 --> It tries 10 different initializations and chooses the best

gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, n_init=10)
gmm.fit(X_scaled)

gmm_hard_labels = gmm.predict(X_scaled)       # Each point is assigned hard cluster assignments, each point assigned to the cluster with highest probability   

gmm_proba = gmm.predict_proba(X_scaled)       # Soft probabilities assigned. It is a matrix with (n_samples, k)

print(f"\n--- GMM Results (n_clusters={k}, covariance_type='{covariance_type}') ---")
print("First 5 Hard Assignments:", gmm_hard_labels[:5])
print("\nFirst 5 Soft Probabilities (Responsibility for each component):\n", gmm_proba[:5])
print("\nLearned Means (Component Centers):\n", gmm.means_)

print("\n--- Visualizing GMM Results ---")
plt.figure(figsize=(10, 7))

# Scatter plot of data points colored by hard assignment
# Using the same color palette as previous plots for consistency
palette = sns.color_palette("viridis", n_colors=k)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=gmm_hard_labels, palette=palette, marker='o', s=50, alpha=0.7, legend='full')


for i in range(k):
    draw_ellipse(gmm.means_[i], gmm.covariances_[i], plt.gca(), alpha=0.2, color=palette[i])




plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=250, label='GMM Means', edgecolors='black')

plt.title(f'GMM Clustering Results (n_components={k}, covariance_type=\'{covariance_type}\')')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend(title='Component')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
