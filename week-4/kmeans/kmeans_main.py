import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmeans_model import KMeansScratch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



#-----------------STEP-1: DATA LOADING AND EDA--------------------------------------------

df = pd.read_csv("./mall_customers.csv")

df.info()                                  # Gives quick overview of the data
print(df.describe())                       # Gives statistical information for all numerical columns
print(df.isnull().sum())                   # checks for missing values in each column


# PLot the Histograms for all numerical distributions of Age, Income and Spending score
plt.figure(figsize=(12, 8))

# Age Distribution
plt.subplot(2, 2, 1)
plt.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution')

# Annual Income Distribution
plt.subplot(2, 2, 2)
plt.hist(df['Annual Income (k$)'], bins=15, color='salmon', edgecolor='black')
plt.title('Annual Income Distribution')

# Spending Score Distribution
plt.subplot(2, 2, 3)
plt.hist(df['Spending Score (1-100)'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()

#Gender distribution
sns.countplot(data=df, x='Gender', palette='pastel')
plt.title('Gender Count')
plt.show()



#-----------------STEP-2: SCALING THE FEATURES--------------------------------------------

features = ['Annual Income (k$)', 'Spending Score (1-100)']   # Selecting only two features to make the 2D plot

X = df[features]                                             # Modifying the dataset to include only two                                         

scaler = StandardScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)

#-----------------STEP-3: RUNNING K-MEANS USING THE CLASS--------------------------------

k = 5                                                                  # Declare the number of clusters needed

k_means_model = KMeansScratch(k=k, random_state = 42)

k_means_model.fit(X_scaled)                                             # Running the model on X_scaled

final_centroids_scaled = k_means_model.centroids_                       # Finding the centroids after fitting the X_scaled data
final_labels = k_means_model.labels_                                    # Finding the labels for each sample in X_scaled
final_inertia = k_means_model.inertia_

#-----------------STEP-4: VISUALIZATION OF RESULTS --------------------------------

print("\n--- Visualizing Clustering Results ---")
plt.figure(figsize=(10, 7))

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=final_labels, palette=sns.color_palette("viridis", n_colors=k), marker='o', s=50, alpha=0.7, legend='full')

plt.scatter(final_centroids_scaled[:, 0], final_centroids_scaled[:, 1], c='red', marker='X', s=250, label='Final Centroids', edgecolors='black')

plt.title(f'K-Means Clustering Results (k={k})')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#-----------------STEP-5: TRANSFORM CENTROIDS BACK TO THE ORIGINAL SCALE --------------------------------

try:
    final_centroids_original = scaler.inverse_transform(final_centroids_scaled)
    print("\nFinal Centroids (Original Scale):")
    # Print centroids with feature names for clarity
    centroid_df = pd.DataFrame(final_centroids_original, columns=features)
    print(centroid_df)
except Exception as e:
    print(f"\nCould not inverse transform centroids: {e}")



#-----------------STEP-6: SKLEARN K-MEANS -------------------- --------------------------------

sklearn_kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
# n_init = number of iterations it will run for intialization and picks the one with least inertia
# By default it is init="k-means++" --> n_init is always 1
# k-means++ gives good results with just one run

sklearn_kmeans.fit(X_scaled)

# Results:

sklearn_centroids_scaled = sklearn_kmeans.cluster_centers_
sklearn_labels = sklearn_kmeans.labels_
sklearn_inertia = sklearn_kmeans.inertia_

print(f"Sklearn KMeans Inertia (k={k}): {sklearn_inertia:.4f}")

#-----------------STEP-7: COMPARE WCSS/INERTIA WITH SCRATCH MODEL AND SKLEARN --------------------------------


print("\n--- WCSS/Inertia Comparison (k=5) ---")
print(f"KMeansScratch WCSS: {final_inertia:.4f}")
print(f"Sklearn KMeans WCSS: {sklearn_inertia:.4f}")


#-----------------STEP-8: VISUALIZE SKLEARN RESULTS --------------------------------

print("\n--- Visualizing Sklearn KMeans Results (k=5) ---")
plt.figure(figsize=(10, 7))

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=sklearn_labels, palette=sns.color_palette("viridis", n_colors=k), marker='o', s=50, alpha=0.7, legend='full')

plt.scatter(sklearn_centroids_scaled[:, 0], sklearn_centroids_scaled[:, 1], c='red', marker='X', s=250, label='Sklearn Centroids', edgecolors='black')

plt.title(f'Sklearn K-Means Clustering Results (k={k})')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#-----------------STEP-9: TRANSFORM SKLEARN CENTROIDS BACK TO THE ORIGINAL SCALE --------------------------------

try:
    sklearn_centroids_original = scaler.inverse_transform(sklearn_centroids_scaled)
    print("\nSKLearn Centroids (Original Scale):")
    # Print centroids with feature names for clarity
    sklearn_centroid_df = pd.DataFrame(sklearn_centroids_original, columns=features)
    print(sklearn_centroid_df)
except Exception as e:
    print(f"\nCould not inverse transform centroids: {e}")



#-----------------STEP-10: ELBOW METHOD FOR OPTIMUM K FINDING --------------------------------

wcss = []                              # List to score WCSS values for each k

k_range = range(1, 11)                 # k to vary from 1 to 10

for k_val in k_range:
    print(f"Running K-Means for k = {k_val}...")

    kmeans = KMeansScratch(k=k_val, random_state=42)

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)


# Plotting WCSS vs k variation

if wcss: # Only plot if WCSS values were successfully collected
    print("\n--- Plotting Elbow Method Results ---")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(k_range) # Ensure all k values are shown on the x-axis
    plt.grid(True)
    plt.show()

else:
    print("\nElbow method plot skipped due to errors collecting WCSS values.")




#-----------------STEP-11: EVALUATE METRICS FOR SCRATCH MODEL AND SKLEARN MODEL --------------------------------

if k > 1:
    try:
        # Silhouette Score: Higher is better
        silhouette_avg_scratch = silhouette_score(X_scaled, final_labels)
        silhouette_avg_sklearn = silhouette_score(X_scaled, sklearn_labels)
        print(f"KMeansScratch Silhouette Score (k={k}): {silhouette_avg_scratch:.4f}")
        print(f"Sklearn Model Silhouette Score (k={k}): {silhouette_avg_sklearn:.4f}")

        # Calinski-Harabasz Index: Higher is better
        calinski_harabasz_avg_scratch = calinski_harabasz_score(X_scaled, final_labels)
        calinski_harabasz_avg_sklearn = calinski_harabasz_score(X_scaled, sklearn_labels)
        print(f"KMeansScratch Calinski-Harabasz Index (k={k}): {calinski_harabasz_avg_scratch:.4f}")
        print(f"Sklearn Model Calinski-Harabasz Index (k={k}): {calinski_harabasz_avg_sklearn:.4f}")

        # Davies-Bouldin Index: Lower is better
        davies_bouldin_avg_scratch = davies_bouldin_score(X_scaled, final_labels)
        davies_bouldin_avg_sklearn = davies_bouldin_score(X_scaled, sklearn_labels)
        print(f"KMeansScratch Davies-Bouldin Index (k={k}): {davies_bouldin_avg_scratch:.4f}")
        print(f"Sklearn Model Davies-Bouldin Index (k={k}): {davies_bouldin_avg_sklearn:.4f}")

    except ValueError as e:
        print(f"Could not calculate metrics for KMeansScratch: {e}")
        
else:
     print("Skipping KMeansScratch metric calculation: Labels not available, k=1, or only one cluster found.")









