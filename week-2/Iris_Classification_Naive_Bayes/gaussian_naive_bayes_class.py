import numpy as np
from scipy.stats import norm

class GaussianNaiveBayes:                              # Without init() you cannot accept constructor parameters, when creating instances
    
    def fit(self, X, y):                       # Function for training of naive bayes model (finding parameters for mean and variances for each feature per class)

        n_samples, n_features = X.shape

        self._classes = np.unique(y)           # The underscore prefix _ is a python convention indicationg this is intended to be protected attribute that shouldn't be accessed outside the class 
        # np.unique finds all unique values in the array y and returns a new sorted array

        n_classes = len(self._classes)         # Total number of unique classes

        # Using dictionaries to store the learned parameters
        # Need to compute, means and variances per feature per class and store them in dictionaries
        self._means = {}                       # {class_label: [mean_feat1, mean_feat2,...]}
        self._variances = {}                   # {class_label: [var_feat1, var_feat2,...]}         
        self._priors = {}                      # Store prior probability for each class   {class_label: prior_prob}     

        for idx, c in enumerate(self._classes):

            X_c = X[y==c]                                   # X_c data contains only for class c

            self._means[c] = X_c.mean(axis = 0)             # calculate mean for each feature for this class only (column means)
            self._variances[c] = X_c.var(axis=0) + 1e-9     # calculate variances fpr each feature for this class only. epsilon is added for numerical stability
            
            self._priors[c] = X_c.shape[0]/float(n_samples)  # P(y=c) = number of samples in class c / total number of samples


    # Also note, self._classes, self_means etc.. are instance attributes. Once it is defined in any method, any other method can be used as long as it is called AFTER the method that defines it

    def predict(self, X):                      # function to predict data points for all new data points

        y_pred = [self.predict_single(x) for x in X]    # for each row in X, run the predict single function to estimate the class for each data point
        return y_pred

    def predict_single(self, x):               # function to predict class for a single sample/datapoint, we will return all posterior probabilities in a list and predict the highest posterior probability class as the y for that sample 
        
        posteriors = []                        # list to store the posterior prob for the data point x

        for idx, c in enumerate(self._classes):

            # the poseterior probability is computed by log(P(y=c)) + Summation of (log(p(xi/y=c))) for all features

            log_prior_c = np.log(self._priors[c])      # log(P(y=c)) for each class

            log_likelihood_c = np.sum(norm.logpdf(x, loc = self._means[c], scale = np.sqrt(self._variances[c])))
            # The above function calculates sum of all the P(xi/y=c). P(x1/y=c) is calculated by probability distribution function with mean of that feature for c class, and the standard deviation of the feature vector for that class c
            # the question is where does each feature of sample x falls from the feature mean in the normal distribution function with a (feature mean and feature standard deviation for that class)
           
            posterior_c = log_prior_c + log_likelihood_c

            posteriors.append(posterior_c)

        
        best_class_index = np.argmax(posteriors)    # FInd the index that has maximum posterior

        return self._classes[best_class_index]      # return the class label 
