import matplotlib.pyplot as plt
from utils.ft_utils import read_csv, get_numeric_columns
from utils.ft_math import ft_mean, ft_std
from utils.constants import BOLD, GREEN, BLUE, END

# Define columns to ignore
ignore_columns = [
    'Index',
    'Hogwarts House',
    'First Name',
    'Last Name',
    'Birthday',
    'Best Hand'
]

def calculate_homogeneity(house_data, course):
    """Calculate homogeneity score for a course"""
    # Get statistics for each house
    house_stats = {}
    for house, data in house_data.items():
        house_stats[house] = {
            'mean': ft_mean(data[course]),
            'std': ft_std(data[course])
        }
    
    # Calculate differences between houses
    means = [stats['mean'] for stats in house_stats.values()]
    stds = [stats['std'] for stats in house_stats.values()]
    
    # Lower differences between means and standard deviations, more homogeneous
    mean_diff = max(means) - min(means)
    std_diff = max(stds) - min(stds)
    
    # Homogeneity score (lower is more homogeneous)
    return mean_diff + std_diff

def create_histograms(data):
    """Create histograms and find the most homogeneous course"""
    
    # Separate data by house
    houses = {}
    for row in data:
        house = row['Hogwarts House']
        if house not in houses:
            houses[house] = []
        houses[house].append(row)

    # Get numeric data for each house
    house_data = {}
    for house, rows in houses.items():
        house_data[house] = get_numeric_columns(rows, ignore_columns)

    # Get list of courses
    courses = list(house_data[list(houses.keys())[0]].keys())
    
    # Calculate homogeneity for each course
    homogeneity_scores = {}
    for course in courses:
        homogeneity_scores[course] = calculate_homogeneity(house_data, course)
    
    # Find the most homogeneous course
    most_homogeneous = min(homogeneity_scores.items(), key=lambda x: x[1])

    sorted_courses = [course for course, _ in sorted(homogeneity_scores.items(), key=lambda x: x[1])]

    # Display histograms
    n_courses = len(sorted_courses)
    n_cols = 4
    n_rows = (n_courses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))
    axes = axes.ravel()
    
    for idx, course in enumerate(sorted_courses):
        ax = axes[idx]
        
        for house, data in house_data.items():
            ax.hist(data[course], bins=20, alpha=0.5, label=house)
        
        # Red bold title if it's the most homogeneous course
        if course == most_homogeneous[0]:
            ax.set_title(f"{course}: {homogeneity_scores[course]:.2f}", fontsize=10, color='red', fontweight='bold')
        else:
            ax.set_title(f"{course}: {homogeneity_scores[course]:.2f}", fontsize=10)
        
        # Set xlabel and ylabel if it's the first histogram
        if idx == 0:
            ax.set_xlabel('Marks')
            ax.set_ylabel('Number of students')
    
    for idx in range(len(sorted_courses), len(axes)):
        fig.delaxes(axes[idx])

    
    print(f"\n{BOLD}Le cours avec la distribution la plus homogène est : {GREEN}{most_homogeneous[0]}{END}")
    print(f"{BOLD}Score d'homogénéité : {BLUE}{most_homogeneous[1]:.2f}{END}")
    
    # Display all scores in table format
    print(f"\n{BOLD}Courses sorted by homogeneity (from most homogeneous to least homogeneous):{END}")
    
    # Define column width
    course_width = max(len(course) for course in sorted_courses)
    score_width = 10
    
    # Table header
    print("-" * course_width + "-+-" + "-" * score_width)
    print(f"{BOLD}{'Cours':<{course_width}} | {'Score':>10}{END}")
    print("-" * course_width + "-+-" + "-" * score_width)
    
    # Table data
    for course, score in sorted(homogeneity_scores.items(), key=lambda x: x[1]):
        print(f"{course:<{course_width}} | {BLUE}{score:>10.2f}{END}")

    # Table footer
    print("-" * course_width + "-+-" + "-" * score_width)

    # Display histograms
    plt.tight_layout()
    plt.show()

def main():
    data = read_csv('datasets/dataset_train.csv')
    create_histograms(data)

if __name__ == "__main__":
    main()