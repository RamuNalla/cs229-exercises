import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#-----------------------------MANUAL GAUSSIAN DESCRIMINAT ANALYSIS -------------------------------------------

np.random.seed(42)

samples_0 = 100
samples_1 = 150

n_samples = samples_0 + samples_1

#class 0 sample distribution
mean_0 = [1, 1]
cov_0 = [[1, 0.5], [0.5, 1]]

#class 1 sample distribution
mean_1 = [5, 5]
cov_1 = [[1.5, -0.3], [-0.3, 0.8]]

X_0 = np.random.multivariate_normal(mean_0, cov_0, samples_0)   # X_0 is a numpy array with shape 100x2 containing data for class_0
X_1 = np.random.multivariate_normal(mean_1, cov_1, samples_1)   

X = np.vstack((X_0, X_1))                                       # Vertically stack the two arrays, therefore the size of X is 250x2
y = np.hstack((np.zeros(samples_0), np.ones(samples_1)))        # Horizontally stacking due to arrays being the size of 1x100 and 1x150

permutation = np.random.permutation(n_samples)                  # random array of indices 0 to 249
X = X[permutation]                                              # reorders the rows of X to the permutation indices
y = y[permutation]                                              # reorders the elements of this vector with the same permutation

print("Random Data generated:")
print(f"  Shape of X: {X.shape}")
print(f"  Shape of y: {y.shape}")
print("-" * 50) 

# Function to calculate gda parameters
def gda_parameters_manual(X, y):
    samples_count, features_count = X.shape                     # rows and columns

    phi = np.mean(y)                                            # Since y is only 0's and 1's, this directly give the proportion of 1's
    
    mu_0 = np.mean(X[y==0], axis = 0)                           # column wise mean for all X's when y is 0. This is still a vector (x1, x2)
    mu_1 = np.mean(X[y==1], axis = 0)                           # column wise mean for all X's when y is 1

    #calculation for sigma matrix
    cov_0_manual = np.zeros((features_count, features_count))   # matrix with 2x2 zeros
    cov_1_manual = np.zeros((features_count, features_count))   

    n_0 = np.sum(y==0)
    n_1 = np.sum(y==1)                                          # count of number of zeroes and ones (we already knew since we declared, but here we are computing everything manually)
                                              
    diff_0 = X[y==0] - mu_0                                     # diff_0 is a vector of 100x2 (we just subtracted every sample with y==0 from its mean)
    cov_0_manual = diff_0.T @ diff_0                            # cov matrix size is 2x2 and we need to do dot product, still we need to divide by number of total samples
                                             
    diff_1 = X[y==1] - mu_1                                 
    cov_1_manual = diff_1.T @ diff_1    
    sigma = (cov_0_manual + cov_1_manual)/samples_count         # As per formula in the lecture

    return phi, mu_0, mu_1, sigma

phi, mu_0, mu_1, sigma = gda_parameters_manual(X, y)

print("Manually calculated GDA Parameters:")
print(f"  phi (P(y=1)): {phi:.4f}")
print(f"  mu_0: {mu_0}")
print(f"  mu_1: {mu_1}")
print(f"  Shared Sigma:\n{sigma}")
print("-" * 50)

# Function to estimate y using gda parameters and new data
def predict_gda(X_new, phi, mu_0, mu_1, sigma):
    
    pdf_1 = multivariate_normal.pdf(X_new, mean = mu_1, cov = sigma)   # Calculate the prob density of new X under the mean of class1 and sigma covariance This gives P{x_new/y=1}
    pdf_0 = multivariate_normal.pdf(X_new, mean = mu_0, cov = sigma)

    #p(y=1/newX) = P(X_new/y=1)*P(y=1)/P(X_new) and to calculate P(X_new), we will do P(X_new/y=1)*P(y=1) + P(X_new/y=0)*P(y=0), remember, we calculted pdf_0 and pdf_1 above

    prob = (pdf_1 * phi)/(pdf_1*phi + pdf_0*(1-phi))                   # Note this is a vector of length of number of samples in X_new 

    predictions = (prob > 0.5).astype(int)

    # There might be that the denominator is very low in the prob calculations creating numerical instability, therefore we can do the follwing
    denominator = pdf_1 * phi + pdf_0 * (1 - phi)
    posterior_prob_1 = np.zeros_like(denominator)                      # creating an array with the length of denominator
    valid_mask = denominator > 1e-15                                   # creates a boolean array with the shape of denominator  [True, True, False, .....]                             

    posterior_prob_1[valid_mask] = (pdf_1[valid_mask] * phi) / denominator[valid_mask]   # It only computes posterior probabilities for inputs where denominator > 1e-15. For other inputs (unsafe), it leaves posterior_prob_1 as 0 (the initialized default)
    predictions = (posterior_prob_1 > 0.5).astype(int)

    return predictions                                                 # returns 0s and 1s with the length of X_new

y_pred = predict_gda(X, phi, mu_0, mu_1, sigma)                        # predict on the trained dataset only

accuracy = np.mean(y_pred == y)

print("Prediction Results on Training Data:")
print(f"  Predictions (first 10): {y_pred[:10]}")
print(f"  Actual labels (first 10): {y[:10].astype(int)}")
print(f"  Accuracy: {accuracy:.4f}")
print("-" * 50)


# Plotting the decision boundary
try: 
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0][y==0], X[:, 1][y==0], c='blue', label='Class 0 (Actual)', alpha=0.6)
    plt.scatter(X[:,0][y==1], X[:, 1][y==1], c='red', label='Class 1 (Actual)', alpha=0.6)

    #creating a mesh_grid for decision boundary

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))    # np.arange(x, y, 0.1) creates a range of numbers from x to y stepping by 0.1. Note that the length of xx1 and xx2 are not same
    ## let's say np.arange(x1_min, x1_max, 0.1) generated m points and np.arange(x2_min, x2_max, 0.1) generated n points the grid will have mxn pints of which xx1 contains first element with mxn size and xx2 contains second element of mxn size

    z = predict_gda(np.c_[xx1.ravel(), xx2.ravel()], phi, mu_0, mu_1, sigma)    # calculates the classes for all z (mxn points)

    z = z.reshape(xx1.shape)    ## opposite of ravel

    # Plot the decision boundary
    plt.contour(xx1, xx2, z, levels=[0.5], colors='black', linestyles='--')

    # z is just 0s and 1s. levels=[0.5] is not referring to values in z that equal 0.5. Instead, it's telling the contour function to draw lines where the values in z transition between less than 0.5 and greater than 0.5.

    plt.title('GDA Decision Boundary with Shared Covariance')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

except ImportError:
    print("Matplotlib not found. Skipping visualization.")

