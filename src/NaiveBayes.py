import numpy as np
import pandas as pd
from typing import Union

class NaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes Classifier.
        
        Parameters:
        -----------
            var_smoothing (float): Portion of the largest variance of all features added to variances for stability.
        """
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = None
        self.class_prior_: np.ndarray = None
        self.theta_: np.ndarray = None
        self.sigma_: np.ndarray = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit Gaussian Naive Bayes according to X, y.
        
        Parameters:
        -----------
            X (DataFrame or ndarray): Training vectors
            y (Series or ndarray): Target values
        
        Returns:
        --------
            self: Fitted estimator
        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[i] = X_c.shape[0] / X.shape[0]
         
            self.theta_[i, :] = X_c.mean(axis=0)

            var = X_c.var(axis=0)

            var_max = np.max(var) if len(var) > 0 else 1.0
            self.sigma_[i, :] = var + self.var_smoothing * var_max

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate joint log likelihood.
        
        Parameters:
        -----------
            X (ndarray): Input samples
        
        Returns:
        --------
            ndarray: Joint log likelihood
        """
        joint_log_likelihood = []
        for i in range(len(self.classes_)):
            jointi = np.log(self.class_prior_[i])

            n_ij = -0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), axis=1)
            
            joint_log_likelihood.append(jointi + n_ij)
        
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        
        Parameters:
        -----------
            X (DataFrame or ndarray): Input samples
        
        Returns:
        --------
            ndarray: Predicted class label for X
        """
        X = X.values if isinstance(X, pd.DataFrame) else X

        jll = self._joint_log_likelihood(X)
        
        return self.classes_[np.argmax(jll, axis=1)]