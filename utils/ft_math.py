def ft_sum(values):
    """ Sum the values of a list """
    total = 0
    for value in values:
        total += value
    return total

def ft_mean(values):
    """ Calculate the mean of a list """
    if not values:
        return 0
    return ft_sum(values) / len(values)

def ft_std(values):
    """ Calculate the standard deviation of a list """
    if len(values) <= 1:
        return 0
    m = ft_mean(values)
    squared_diff_sum = ft_sum((x - m) ** 2 for x in values)
    return ft_sqrt(squared_diff_sum / (len(values) - 1))

def ft_percentile(values, p):
    """ Calculate the percentile of a list.

    p is the percentile to calculate
    """
    if not values:
        return 0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_values):
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    return sorted_values[f]

def ft_median(values):
    """ Calculate the median of a list """
    return ft_percentile(values, 0.5)

def ft_quartile(values, q):
    """ Calculate the quartile of a list.
    
    q is the quartile to calculate
    """
    return ft_percentile(values, q)

def ft_min(values):
    """ Calculate the minimum value of a list """
    if not values:
        return 0
    result = values[0]
    for value in values:
        if value < result:
            result = value
    return result

def ft_max(values):
    """ Calculate the maximum value of a list """
    if not values:
        return 0
    result = values[0]
    for value in values:
        if value > result:
            result = value
    return result

def ft_count(values):
    """ Calculate the count of a list """
    return len(values)

def ft_product(values):
    """ Calculate the product of a list """
    if not values:
        return 0
    result = 1
    for value in values:
        result *= value
    return result

def ft_cumsum(values):
    """ Calculate the cumulative sum of a list """
    return [ft_sum(values[:i+1]) for i in range(len(values))]

def ft_sqrt(x):
    """ Calculate the square root of a number """
    if x < 0:
        raise ValueError("math domain error")
    if x == 0:
        return 0
    
    # Use Newton's method to calculate the square root
    guess = x / 2
    epsilon = 1e-9  # Desired precision
    
    while True:
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < epsilon:
            return new_guess
        guess = new_guess

def ft_minmax(values):
    """Normalize the data between 0 and 1"""
    values = list(values)  # Convert to list
    if not values:
        return []
    min_val = ft_min(values)
    max_val = ft_max(values)
    if max_val == min_val:
        return [0.5 for _ in values]
    return [(x - min_val) / (max_val - min_val) for x in values]
