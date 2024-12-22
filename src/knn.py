import pickle
import numpy as np

class KNN:
    def __init__(self, k=3, metric='euclidean', p=2):
    # Initialize KNN classifier
    
    # k (int): Number of neighbors
    # metric (str): Distance metric ('euclidean', 'manhattan', or 'minkowski')
    # p (int): Power parameter for Minkowski metric
    
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def _euclidean_distance(self, x1, x2):
        #Calculate Euclidean distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        #Calculate Manhattan distance between two points
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1, x2):
        #Calculate Minkowski distance between two points
        return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)

    def _calculate_distance(self, x1, x2):
        #Calculate distance based on chosen metric
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError("Invalid distance metric. Choose 'euclidean', 'manhattan', or 'minkowski'")

    def fit(self, X, y):
    # Fit the KNN model
    # Parameters:
    # X (array-like): Training data
    # y (array-like): Target values
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        # Predict class labels for samples in X
        
        # Parameters:
        # X (array-like): Test samples
        
        # Returns:
        # array: Predicted class labels
        X = np.array(X)
        predictions = []

        for x in X:
            # Calculate distances to all training samples
            distances = []
            for x_train in self.X_train:
                distance = self._calculate_distance(x, x_train)
                distances.append(distance)

            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]

            # Majority vote
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)

    def save_model(self, filename):
        #Save the model to a file
        # Parameters:
        # filename (str): Path to save the model
       
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
      
        # Load a saved model from a file
        
        # Parameters:
        # filename (str): Path to the saved model
        
        # Returns:
        # KNN: Loaded model
    
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(k=model_data['k'], metric=model_data['metric'], p=model_data['p'])
        model.X_train = model_data['X_train']
        model.y_train = model_data['y_train']
        return model