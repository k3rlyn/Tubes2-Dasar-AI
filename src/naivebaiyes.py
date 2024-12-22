import numpy as np
import pickle

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = len(X_c) / len(X)

        return self

    def _calculate_likelihood(self, x, mean, var):
        # Add small constant to variance to prevent division by zero
        eps = 1e-10
        coef = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = -0.5 * ((x - mean) ** 2) / (var + eps)
        return coef * np.exp(exponent)

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        
        # For each sample
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(
                    self._calculate_likelihood(x, self.mean[idx, :], self.var[idx, :])
                ))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Select class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)

    def save_model(self, filename):
        model_data = {
            'classes': self.classes,
            'mean': self.mean,
            'var': self.var,
            'priors': self.priors
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.classes = model_data['classes']
        model.mean = model_data['mean']
        model.var = model_data['var']
        model.priors = model_data['priors']
        return model