import numpy as np
from utils.ft_utils import read_csv, is_numeric
from utils.ft_math import ft_minmax
from utils.constants import BOLD, BLUE, END

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def cost_function(X, y, theta):
#     m = len(y)
#     h = sigmoid(X @ theta)
#     epsilon = 1e-5
#     cost = (1/m) * ((-y).T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
#     return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * gradient
    return theta

def train_logistic_regression(X, y, alpha=0.01, num_iters=1000):
    X = np.insert(X, 0, 1, axis=1)  # Add intercept term
    theta = np.zeros(X.shape[1])
    theta = gradient_descent(X, y, theta, alpha, num_iters)
    return theta

def train_one_vs_all(X, y, num_classes, alpha=0.01, num_iters=1000):
    """Train one model per class using one-vs-all strategy"""
    n_features = X.shape[1]
    all_theta = np.zeros((num_classes, n_features + 1))
    
    for i in range(num_classes):
        # Prepare binary labels (1 for current class, 0 for others)
        binary_y = (y == i).astype(int)
        # Train model for current class
        all_theta[i] = train_logistic_regression(X, binary_y, alpha, num_iters)
    
    return all_theta

def standardize_data(X):
    """Standardize features to have mean=0 and std=1"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def prepare_data(file_path, features, target):
    data = read_csv(file_path)
    
    # Filter rows with non-numeric values
    filtered_data = []
    for row in data:
        if all(is_numeric(row[feature]) for feature in features):
            filtered_data.append(row)
    
    X = np.array([[float(row[feature]) for feature in features] for row in filtered_data])
    
    # Standardize instead of normalize
    X, mean, std = standardize_data(X)
    
    # Convert labels to numeric
    house_mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    y = np.array([house_mapping[row[target]] for row in filtered_data])
    
    return X, y, mean, std

def main():
    # Path to the training file
    train_file = 'datasets/dataset_train.csv'
    
    # Features and target
    features = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']
    target = 'Hogwarts House'
    
    # Prepare the data
    X, y, mean, std = prepare_data(train_file, features, target)
    
    # Train one model per class
    num_classes = 4  # Gryffindor, Hufflepuff, Ravenclaw, Slytherin
    all_theta = train_one_vs_all(X, y, num_classes)
    
    # Save the weights and scaling parameters
    np.savez('model.npz', theta=all_theta, mean=mean, std=std)
    
    print("\nTraining completed and saved in model.npz\n")
    print(f"{BOLD}Theta shape: {BLUE}{all_theta.shape}{END}")

if __name__ == "__main__":
    main()