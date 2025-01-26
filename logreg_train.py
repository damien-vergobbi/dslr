import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import BOLD, END

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=500):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        # Handle large negative values
        negative_mask = z < 0
        positive_mask = ~negative_mask
        result = np.zeros_like(z)
        
        # For positive values: 1 / (1 + exp(-z))
        result[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        # For negative values: exp(z) / (1 + exp(z))
        exp_z = np.exp(z[negative_mask])
        result[negative_mask] = exp_z / (1 + exp_z)
        
        return result

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for _ in range(self.iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Gradient descent
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculate and store loss
            loss = -np.mean(y * np.log(predictions + 1e-15) + 
                          (1-y) * np.log(1-predictions + 1e-15))
            self.loss_history.append(loss)

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

def preprocess_data(df):
    # Select features based on analysis
    features = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']
    X = df[features].fillna(df[features].mean())
    return X

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def train_models():
    # Load data
    df = pd.read_csv('./datasets/dataset_train.csv')
    X = preprocess_data(df)
    
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights_data = {}
    accuracies = {}

    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    # Train models and store weights directly
    for idx, house in enumerate(houses):
        y = (df['Hogwarts House'] == house).astype(int)
        model = LogisticRegression()
        model.fit(X.values, y)

        # Plot in corresponding subplot
        axes[idx].plot(model.loss_history)
        axes[idx].set_title(f'{house}')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')

        # Store weights and calculate accuracy
        y_pred = model.predict_proba(X.values) >= 0.5
        acc = calculate_accuracy(y, y_pred)
        accuracies[house] = acc
        print(f"{BOLD}{house}{END} accuracy: {BOLD}{acc:.4f}{END}")

        weights_data[house] = {
            'weights': ','.join(map(str, model.weights)),
            'bias': model.bias
        }
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Convert to DataFrame and save
    weights_df = pd.DataFrame.from_dict(weights_data, orient='index')
    weights_df.to_csv('weights.csv', index=True, index_label='house')

if __name__ == "__main__":
    train_models()