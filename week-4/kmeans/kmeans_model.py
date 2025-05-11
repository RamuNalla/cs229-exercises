import numpy as np
from scipy.spatial.distance import cdist
import warnings

class KMeansScratch:

    # Instance Attributes
    # k: The number of clusters
    # max_iter: Maximum number of iterations
    # tol: Tolerance for centroid movement to declare convergence

    def __init__(self, k=5, max_iter=300, tol=1e-4, random_state = None):     # Default values
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tol, float) or tol < 0:
            raise ValueError("tol must be a non-negative float.")
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # centroids_: Final cluster centroids (k, n_features)
        # labels_ : cluster labels assigned to each data point (n_samples,)
        # inertia_ : sum of squared dustances of samples to closest cluster center
        # n_iter_: Number of iterations run
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter = 0


    def _initialize_centroids(self, X):           # Initialize centroids by randomly choosing k_data points from X

        if self.random_state is not None:
            np.random.seed(self.random_state)     # Set seed for reproducibility

        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError(f"Number of clusters k ({self.k}) cannot be greater than the number of samples ({n_samples}).")
        
        random_indices = np.random.choice(n_samples, self.k, replace=False)  # Select k unique random indices from n_samples
        # random_indices is an array
        self.centroids_ = X[random_indices].astype(float)                    # self.centroids (k, n_features)
        #print(f"Initialized {self.k} random centroids.")

    # Assign clusters to each of the sample based on distance computation of each sample with k clusters. Find the min distance cluster and store the cluster labels in the cluster_assignments array
    def _assign_clusters(self, X):

        # X has a shape of (n_samples, n_features)
        # centroids (k, n_features)
        # Need distances in the shape of (n_samples, k), distance[i][j] gives the distance of samples i to centroid j
        
        # cdist function calculates pairwaise distances between each row in two 2D arrays

        distances = cdist(X, self.centroids_, metric = 'euclidean')   # --> shape: (n_samples, k)

        cluster_assignments = np.argmin(distances, axis=1)            # For each row, find the index of the minimum value
        return cluster_assignments                                    # shape will be (n_samples, )

    # Update the centroid to be the mean of the points assigned to each cluster
    def _update_centroids(self, X, labels):
        
        if self.centroids_ is None:
             raise RuntimeError("Centroids not initialized.")

        new_centroids = np.zeros_like(self.centroids_)

        for i in range(self.k):

            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis = 0)
                                                                                                # Handle empty clusters
            else:     
                warnings.warn(f"Cluster {i} became empty during update. Keeping old centroid.", UserWarning)
                new_centroids[i] = self.centroids_[i] 

        return new_centroids
    

    # Calculate the sum of squared distances of samples to their closest cluster  (WCSS Value for the resulting cluster)
    def _calculate_inertia(self, X, labels):

        inertia = 0.0
        for i in range(self.k):
            cluster_points = X[labels==i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids_[i])**2)
        
        return inertia

    # Computes K-means clustering by iterating assign clusters and update centroids until convergence
    def fit(self, X):

        self._initialize_centroids(X)

        for i in range(self.max_iter):
            self.n_iter = i+1
            old_centroids = self.centroids_.copy()                             # Store old centroids for convergence check

            self.labels_ = self._assign_clusters(X)                            # Assign clusters  

            self.centroids_ = self._update_centroids(X, self.labels_)          # Update centroids

            centroid_shift = np.linalg.norm(self.centroids_ - old_centroids)   # Sum of squares of all elements in the matrix
            
            if centroid_shift < self.tol:
                #print(f"Converged after {self.n_iter} iterations.")
                break

        else:                                                                 # The else executes if the loop completes without the break        
            warnings.warn(f"K-Means did not converge within {self.max_iter} iterations. The final solution might not be optimal.", UserWarning)


        self.inertia_ = self._calculate_inertia(X, self.labels_)
        print(f"Final Inertia: {self.inertia_:.4f}")

        return self                                                           # useful for chaining when called fit method followed by any other method in a single line of code


    # To predict the closest cluster each sample in X belongs to
    def predict(self, X):
        
        # Possible errors
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input data X must be a 2D NumPy array.")
        if X.shape[1] != self.centroids_.shape[1]:
             raise ValueError(f"Input data has {X.shape[1]} features, but model was fitted with {self.centroids_.shape[1]} features.")

        return self._assign_clusters(X)                                       # Returns an array of cluster assignments with (n_samples,)



