import numpy as np
import pandas as pd

def parse_weights(weight_str):
    # Parse comma-separated string of weights
    return np.array([float(x) for x in weight_str.split(',')])

def sigmoid(z):
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

def predict_house(X, weights_df):
    predictions = {}
    
    for house in weights_df.index:
        # Get weights and bias
        weights = parse_weights(weights_df.loc[house, 'weights'])
        bias = float(weights_df.loc[house, 'bias'])
        
        # Calculate predictions
        linear_pred = np.dot(X, weights) + bias
        predictions[house] = sigmoid(linear_pred)
    
    return pd.DataFrame(predictions).idxmax(axis=1)

def main():
    # Load data
    test_df = pd.read_csv('./datasets/dataset_test.csv')
    weights_df = pd.read_csv('weights.csv', index_col=0)
    
    # Preprocess features
    features = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']
    X = test_df[features].fillna(test_df[features].mean())
    
    # Make predictions
    predictions = predict_house(X.values, weights_df)
    
    # Save results
    result_df = pd.DataFrame({
        'Index': test_df.index,
        'Hogwarts House': predictions
    })
    result_df.to_csv('houses.csv', index=False)

if __name__ == "__main__":
    main()