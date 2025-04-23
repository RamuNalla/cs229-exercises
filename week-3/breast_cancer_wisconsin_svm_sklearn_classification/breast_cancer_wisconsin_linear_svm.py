import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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



# Function to plot the decision boundary, margins and support vectors for a trained linear SVM model
def plot_svm_boundary(svm_model, X, y, title = "SVM Decision Boundary and Margins"):

    plt.figure(figsize=(8, 6))
    
    # Plot the data points with blue for 1 and green for 0
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', edgecolors='k')    #c=y means set the color of each point based on label 'y', s=50 means size of each point in the scatter plot


    # Plotting the decision boundary with decision function, 
    # Decision function gives the signed distance from each point to the distance boundary, Z = 0, point lies on the decision boundary
    # Z = +1, point lies exactly 1 unit away (in the feature space) on the positive class side
    # The Z = +1/-1 points are edges of the margin and the band the SVM maximises, ideally in a perfectly separable dataset, there should be no data points that should fall inside the margin
    # The values of decision function is 0 for decision boundary
    ax = plt.gca()                   # Get current axes
    xlim = ax.get_xlim()             # xlim is an array with min and max limits in an array
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)                                   # makes a matrix of 50x50 datapoints

    xy = np.vstack([XX.ravel(), YY.ravel()]).T                     # XX. ravel will give an array of (2500,) vertical stacking it makes it (2,2500). Transpose makes it (2500,2)

    Z = svm_model.decision_function(xy).reshape(XX.shape)          # decision_function returns signed distance of each point from the decision boundary, so this will be (2500,) array. Now we will reshape it into (50x50), which is the size of XX

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])    # contour lines with Z = -1, 0, 1, Z actually contains signed distance, therefore floating point values. We want contour lines only for -1, 0, 1, therefore, 3 levels are mentioned in the contour plot


    # Support vectors are the data points exactly on Z = +1 and -1. 
    # No data point should lie within the margin because of the hard-margin SVM. All SVs should lie on the margin
    # In the plots shown, some data points lies within the margin due to Soft-Margin SVM
    
    # Support vectors include:
    # Points exactly on the margin (Z = ±1).
    # And also points inside the margin (|Z| < 1) or even on the wrong side (Z has the wrong sign) in the soft-margin setting.

    # C controls the trade-off between margin size and classification error.
    # Smaller C → larger margin, more violations allowed.
    # Larger C → stricter margin, fewer violations allowed. Note in the code below C = 1000


    support_vectors = svm_model.support_vectors_                   # Get the support vectors from the trained model, the size of support_vectors is (n_support, 2) {2D dataset}

    ax.scatter(support_vectors[:,0], support_vectors[:,1], s=100, linewidth=1, facecolors="none", edgecolors='k', label='Support Vectors')

    plt.title(title)
    plt.xlabel('Radius mean')
    plt.ylabel('Texture mean')
    plt.legend()
    ax.set_xlim(xlim) # Restore original limits
    ax.set_ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



# Feature Scaling (Avoiding Data leakage)
scaler = StandardScaler()                               # Features are stabdardized so that they have 0 mean and 1 standard deviation

X_train_scaled = scaler.fit_transform(X_train)          # These are numpy arrays after standard scaler
X_test_scaled = scaler.transform(X_test)                # We don't want the model to have any info about the test data during training. All preprocessing steps should fit only on training data and then applied to the test data

# Training the model

svm_model = SVC(kernel='linear', C=1000)    
# kerner = linear --> the model will try to find linear decision boundary
# c = 1000 --> regularization parameter (soft margin parameter), it controls trade-off between maximizing the margin and minimizing the classification error
# Large value of C means the model will try to classify all training examples correctly (it might result in smaller margib, overfitting risk)
# Smaller C means model will allow some misclassifications to achieve a wider margin

svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Plotting both SVM decision boundaries on test and trained data

plot_svm_boundary(svm_model, X_train_scaled, y_train.values, title="Train Data (SVM)")

plot_svm_boundary(svm_model, X_test_scaled, y_test.values, title="Test Data (SVM)")





