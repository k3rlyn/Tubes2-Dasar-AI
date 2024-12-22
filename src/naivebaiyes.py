import numpy as np
import pickle
from collections import defaultdict

class GaussianNB:
    def __init__(self):
        """
        Initialize Gaussian Naive Bayes classifier
        """
        self.classes = None  # Unique classes in the target
        self.class_priors = {}  # Prior probabilities of classes P(y)
        self.means = defaultdict(dict)  # Mean of each feature per class
        self.variances = defaultdict(dict)  # Variance of each feature per class
        self.n_features = None  # Number of features
        self.epsilon = 1e-10  # Small value to prevent division by zero

    def _calculate_prior_probabilities(self, y):
        """
        Calculate prior probabilities P(y) for each class
        
        Parameters:
        y (array-like): Target values
        """
        total_samples = len(y)
        for class_label in self.classes:
            class_count = np.sum(y == class_label)
            self.class_priors[class_label] = class_count / total_samples

    def _calculate_mean_variance(self, X, y):
        """
        Calculate mean and variance for each feature in each class
        
        Parameters:
        X (array-like): Training features
        y (array-like): Target values
        """
        for class_label in self.classes:
            # Get samples for current class
            class_samples = X[y == class_label]
            
            # Calculate mean and variance for each feature
            self.means[class_label] = np.mean(class_samples, axis=0)
            self.variances[class_label] = np.var(class_samples, axis=0) + self.epsilon

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes model
        
        Parameters:
        X (array-like): Training features
        y (array-like): Target values
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        
        # Calculate prior probabilities
        self._calculate_prior_probabilities(y)
        
        # Calculate mean and variance for each feature in each class
        self._calculate_mean_variance(X, y)

    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate likelihood P(x|y) using Gaussian probability density function
        
        Parameters:
        x (float): Feature value
        mean (float): Mean of the feature for a class
        var (float): Variance of the feature for a class
        
        Returns:
        float: Likelihood probability
        """
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)

    def _predict_single(self, x):
        """
        Predict class for a single sample
        
        Parameters:
        x (array-like): Single sample features
        
        Returns:
        int: Predicted class
        """
        posteriors = {}
        
        # Calculate posterior probability for each class
        for class_label in self.classes:
            # Start with prior probability
            prior = np.log(self.class_priors[class_label])
            
            # Calculate likelihood for each feature
            likelihood = np.sum([
                np.log(self._calculate_likelihood(
                    x[feature_idx],
                    self.means[class_label][feature_idx],
                    self.variances[class_label][feature_idx]
                ))
                for feature_idx in range(self.n_features)
            ])
            
            # Posterior is prior + sum of log likelihoods
            posteriors[class_label] = prior + likelihood
        
        # Return class with highest posterior probability
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X (array-like): Test samples
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        X (array-like): Test samples
        
        Returns:
        array: Class probabilities for each sample
        """
        X = np.array(X)
        probas = []
        
        for x in X:
            # Calculate unnormalized log probabilities
            log_probs = {}
            for class_label in self.classes:
                prior = np.log(self.class_priors[class_label])
                likelihood = np.sum([
                    np.log(self._calculate_likelihood(
                        x[feature_idx],
                        self.means[class_label][feature_idx],
                        self.variances[class_label][feature_idx]
                    ))
                    for feature_idx in range(self.n_features)
                ])
                log_probs[class_label] = prior + likelihood
            
            # Convert log probabilities to probabilities and normalize
            max_log_prob = max(log_probs.values())
            probs = {
                class_label: np.exp(log_prob - max_log_prob)
                for class_label, log_prob in log_probs.items()
            }
            normalizer = sum(probs.values())
            probs = {k: v/normalizer for k, v in probs.items()}
            
            # Sort probabilities by class label
            sorted_probs = [probs[class_label] for class_label in sorted(self.classes)]
            probas.append(sorted_probs)
        
        return np.array(probas)

    def save_model(self, filename):
        """
        Save the model to a file
        
        Parameters:
        filename (str): Path to save the model
        """
        model_data = {
            'classes': self.classes,
            'class_priors': self.class_priors,
            'means': dict(self.means),
            'variances': dict(self.variances),
            'n_features': self.n_features,
            'epsilon': self.epsilon
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        """
        Load a saved model from a file
        
        Parameters:
        filename (str): Path to the saved model
        
        Returns:
        GaussianNB: Loaded model
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.classes = model_data['classes']
        model.class_priors = model_data['class_priors']
        model.means = defaultdict(dict, model_data['means'])
        model.variances = defaultdict(dict, model_data['variances'])
        model.n_features = model_data['n_features']
        model.epsilon = model_data['epsilon']
        
        return model