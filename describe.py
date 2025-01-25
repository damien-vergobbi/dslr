import sys
from utils.ft_math import ft_mean, ft_std, ft_median, ft_quartile, ft_min, ft_max, ft_count
from utils.ft_utils import read_csv, get_numeric_columns

ignore_columns = [
    'Index',
    'Hogwarts House',
    'First Name',
    'Last Name',
    'Birthday',
    'Best Hand'
]

metrics = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

def describe_data(numeric_data):
    """ Describe the data """
    stats = {}
    for feature, values in numeric_data.items():
        stats[feature] = {
            'Count': ft_count(values),
            'Mean': ft_mean(values),
            'Std': ft_std(values),
            'Min': ft_min(values),
            '25%': ft_quartile(values, 0.25),
            '50%': ft_median(values),
            '75%': ft_quartile(values, 0.75),
            'Max': ft_max(values)
        }
    return stats

def display_stats(stats):
    """ Display the statistics """
    if not stats:
        print("No numeric data found")
        return
        
    # Set a fixed width for each column
    col_width = 25
    
    # Display the headers
    features = list(stats.keys())
    print(f"{'':>{col_width}}", end='')
    for feature in features:
        # Truncate long names
        feature_name = feature[:col_width-1]
        print(f"{feature_name:>{col_width}}", end='')
    print()

    # Display the statistics
    for metric in metrics:
        print(f"{metric:>{col_width}}", end='')
        for feature in features:
            value = stats[feature][metric]
            if metric == 'Count':
                print(f"{value:>{col_width}.0f}", end='')
            else:
                print(f"{value:>{col_width}.6f}", end='')
        print()

def main():
    """ Main function """

    # Check if the user provided a filename
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)
        
    # Get the filename from the command line
    filename = sys.argv[1]
    data = read_csv(filename)

    # Get the numeric columns without the ignore_columns
    numeric_data = get_numeric_columns(data, ignore_columns)
    stats = describe_data(numeric_data)
    
    # Print stats
    display_stats(stats)
    
    # print("\n=== Pandas (Verification only) ===")
    # import pandas as pd
    # df = pd.DataFrame(data)
    # df = df.drop(columns=ignore_columns)
    
    # # Convertir les colonnes en type numérique
    # for col in df.columns:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # print(df.describe())

if __name__ == "__main__":
    main()
