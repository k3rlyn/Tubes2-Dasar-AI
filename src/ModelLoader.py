import pickle

class ModelLoader:
    @staticmethod
    def save(model, filename):
        """
        Save model to file
        
        Parameters:
        -----------
        model : object
            Model to save
        filename : str
            File path to save the model
        """
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(filename):
        """
        Load model from file
        
        Parameters:
        -----------
        filename : str
            File path to load the model from
            
        Returns:
        --------
        object
            Loaded model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)