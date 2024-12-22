import numpy as np
from collections import Counter
import pickle

class KNN:
    def __init__(self, k=5, metric='euclidean', p=2):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _calculate_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _get_neighbors(self, x):
        # Calculate distances to all training points
        distances = np.array([self._calculate_distance(x, x_train) 
                            for x_train in self.X_train])
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Return labels of k nearest neighbors
        return self.y_train[k_indices]

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        # Make prediction for each sample
        for x in X:
            # Get k nearest neighbors
            neighbors = self._get_neighbors(x)
            
            # Majority vote
            most_common = Counter(neighbors).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)

    def save_model(self, filename):
        """Save model to file"""
        model_data = {
            'k': self.k,
            'metric': self.metric,
            'p': self.p,
            'X_train': self.X_train,
            'y_train': self.y_train
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            k=model_data['k'],
            metric=model_data['metric'],
            p=model_data['p']
        )
        model.X_train = model_data['X_train']
        model.y_train = model_data['y_train']
        return model