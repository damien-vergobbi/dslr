import matplotlib.pyplot as plt
from utils.ft_utils import read_csv, get_numeric_columns
from utils.ft_math import ft_minmax
from utils.constants import BOLD, GREEN, BLUE, END

ignore_columns = [
    'Index',
    'Hogwarts House',
    'First Name',
    'Last Name',
    'Birthday',
    'Best Hand'
]

def calculate_similarity(data1, data2):
    """Calculate the similarity between two series of data"""
    # Filter valid data (non None/NaN)
    valid_pairs = [(x, y) for x, y in zip(data1, data2) if x is not None and y is not None]
    if not valid_pairs:
        return float('inf')
    
    data1_clean = [x for x, _ in valid_pairs]
    data2_clean = [y for _, y in valid_pairs]
    
    # Normalize data
    norm1 = ft_minmax(data1_clean)
    norm2 = ft_minmax(data2_clean)
    
    # Calculate the average difference
    diff_sum = sum(abs(n1 - n2) for n1, n2 in zip(norm1, norm2))
    return diff_sum / len(norm1)

def find_similar_features(data):
    """Find the two most similar features"""
    numeric_data = get_numeric_columns(data, ignore_columns)
    
    # Calculate the similarity between each pair of features
    similarities = {}
    features = list(numeric_data.keys())
    
    for i, feat1 in enumerate(features):
        for feat2 in features[i+1:]:
            pair = f"{feat1} vs {feat2}"
            similarity = calculate_similarity(numeric_data[feat1], numeric_data[feat2])
            similarities[pair] = similarity
    
    # Find the most similar pair
    most_similar = min(similarities.items(), key=lambda x: x[1])
    
    return similarities, most_similar

def create_scatter_plot(data, feature1, feature2):
    """Create a scatter plot for two features"""
    numeric_data = get_numeric_columns(data, ignore_columns)
    
    # Filter valid data
    valid_pairs = [(x, y) for x, y in zip(numeric_data[feature1], numeric_data[feature2]) 
                  if x is not None and y is not None]
    
    if not valid_pairs:
        print(f"No valid data for {feature1} vs {feature2}")
        return
    
    # Split data into x and y
    x_data = [x for x, _ in valid_pairs]
    y_data = [y for _, y in valid_pairs]
    
    # Normalize data
    x_norm = ft_minmax(x_data)
    y_norm = ft_minmax(y_data)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_norm, y_norm, alpha=0.5)
    
    plt.title(f'Scatter Plot: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    plt.tight_layout()
    plt.show()

def main():
    # Read data
    data = read_csv('datasets/dataset_train.csv')
    
    # Find similar features
    similarities, most_similar = find_similar_features(data)
    
    # Display results
    print(f"{BOLD}The two most similar features are:{END}")
    feature1, feature2 = most_similar[0].split(" vs ")
    print(f"{GREEN}{feature1}{END} and {GREEN}{feature2}{END}")
    print(f"{BOLD}Similarity score: {BLUE}{most_similar[1]:.4f}{END}")
    
    # Display similarity table
    print(f"\n{BOLD}Similarity ranking:{END}")
    
    # Calculate column widths
    pair_width = max(len(pair) for pair in similarities.keys())
    score_width = 10
    
    # Table header
    print("-" * pair_width + "-+-" + "-" * score_width)
    print(f"{BOLD}{'Pairs':<{pair_width}} | {'Score':>10}{END}")
    print("-" * pair_width + "-+-" + "-" * score_width)
    
    # Table data
    for pair, score in sorted(similarities.items(), key=lambda x: x[1]):
        print(f"{pair:<{pair_width}} | {BLUE}{score:>10.4f}{END}")
    
    # Table footer
    print("-" * pair_width + "-+-" + "-" * score_width)
    
    # Create scatter plot
    create_scatter_plot(data, feature1, feature2)

    # For testing other pairs
    # Get pair Herbology vs Potions
    # create_scatter_plot(data, 'Herbology', 'Potions')

if __name__ == "__main__":
    main()