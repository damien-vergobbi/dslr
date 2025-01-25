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
    'Best Hand',
    # Removed because too homogenous
    'Care of Magical Creatures',
    'Arithmancy',
    # Removed because not relevant
    'Defense Against the Dark Arts', #(same as Astronomy)
]

def create_pair_plot(data):
    """Create a pair plot for all numeric features"""
    # Get numeric data
    numeric_data = get_numeric_columns(data, ignore_columns)
    features = list(numeric_data.keys())
    
    # Create subplots grid
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(20, 20))
    
    # Define colors for each house
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }
    
    # For each pair of features
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            ax = axes[i, j]
            
            # For each house
            for house in colors:
                # Filter data for this house
                house_data = [row for row in data if row['Hogwarts House'] == house]
                
                # Get and clean data
                valid_pairs = []
                for row in house_data:
                    try:
                        x = float(row[feat1]) if row[feat1] is not None else None
                        y = float(row[feat2]) if row[feat2] is not None else None
                        if x is not None and y is not None:
                            valid_pairs.append((x, y))
                    except (ValueError, TypeError):
                        continue
                
                if not valid_pairs:
                    continue
                    
                x_data = [x for x, _ in valid_pairs]
                y_data = [y for _, y in valid_pairs]
                
                # Normalize data
                x_norm = ft_minmax(x_data)
                y_norm = ft_minmax(y_data)
                
                # Create scatter plot for this house
                ax.scatter(x_norm, y_norm, alpha=0.5, s=1, c=colors[house], label=house)
            
            # Configure axes
            if i == n-1:  # Last line
                ax.set_xlabel(feat2, fontsize=8)
            if j == 0:    # First column
                ax.set_ylabel(feat1, fontsize=8)
            
            ax.tick_params(labelsize=6)
            
            # Add legend only for the first graph
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    
    plt.tight_layout()
    plt.show()

def main():
    # Read data
    data = read_csv('datasets/dataset_train.csv')
    
    print(f"\n{BOLD}Creation of pair plot to visualize relations between features{END}")
    print(f"{BOLD}Each point represents a student{END}")
    print(f"{BOLD}Data is normalized between 0 and 1{END}")
    
    print(f"\n{BOLD}Analyse :{END}")
    print("1. Diagonals show the distribution of each feature")
    print("2. Other cases show the relations between pairs of features")
    print("3. A diagonal shape indicates a positive correlation")
    print("4. An inverse shape indicates a negative correlation")
    print("5. A dispersed cloud indicates little or no correlation")
    
    create_pair_plot(data)

if __name__ == "__main__":
    main()