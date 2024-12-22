import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class KNN:
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean'):
        """
        Initialize KNN with more flexible distance metric options.
        
        Parameters:
        -----------
            k (int, optional): Number of nearest neighbors to consider (default=3).
            distance_metric (str, optional): Distance metric to use for neighbor calculation (default='euclidean', 'manhattan', 'minkowski').
        """
        self.k = k
        if distance_metric == "manhattan":
            self.distance_metric = "cityblock"
        else:
            self.distance_metric = distance_metric
    
    def _ensure_ndarray(self, X):
        """
        Ensure the input data is converted to a NumPy ndarray.
        
        Parameters:
        -----------
            X (pd.DataFrame, csr_matrix, or ndarray): Input data to convert.
        
        Returns:
        --------
            np.ndarray: Converted data as an ndarray.
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, csr_matrix):
            return X.toarray()
        elif isinstance(X, np.ndarray):
            return X
        if isinstance(X, pd.Series):
            return X.values
        else:
            raise TypeError("Input data must be a DataFrame, csr_matrix, or ndarray.")
    
    def fit(self, X_train, y_train):
        """
        Store training data after ensuring it's an ndarray.
        
        Parameters:
        -----------
            X_train (pd.DataFrame, csr_matrix, or ndarray): Feature matrix for training.
            y_train (pd.Series, or ndarray): Target labels for training.
        """
        self.X_train = self._ensure_ndarray(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test) -> np.ndarray:
        """
        Predict labels for test points in a memory-efficient manner.
        
        Parameters:
        -----------
            X_test (pd.DataFrame, csr_matrix, or ndarray): Feature matrix for testing
        
        Returns:
        --------
            np.ndarray: Predicted labels for test points
        """
        X_test = self._ensure_ndarray(X_test)
        predictions = []
        
        for test_point in X_test:
            distances = cdist([test_point], self.X_train, metric=self.distance_metric)
            
            k_indices = np.argsort(distances[0])[:self.k]
            
            k_nearest_labels = self.y_train[k_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            
            predictions.append(most_common_label)
        
        return np.array(predictions)