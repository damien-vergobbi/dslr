def sum_values(values):
    """ Sum the values of a list """
    total = 0
    for value in values:
        total += value
    return total

def mean(values):
    """ Calculate the mean of a list """
    if not values:
        return 0
    return sum_values(values) / len(values)

def std(values):
    """ Calculate the standard deviation of a list """
    if len(values) <= 1:
        return 0
    m = mean(values)
    squared_diff_sum = sum_values((x - m) ** 2 for x in values)
    return sqrt(squared_diff_sum / (len(values) - 1))

def percentile(values, p):
    """ Calculate the percentile of a list """
    if not values:
        return 0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_values):
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    return sorted_values[f]

def median(values):
    """ Calculate the median of a list """
    return percentile(values, 0.5)

def quartile(values, q):
    """ Calculate the quartile of a list """
    return percentile(values, q / 4)

def min_value(values):
    """ Calculate the minimum value of a list """
    if not values:
        return 0
    result = values[0]
    for value in values:
        if value < result:
            result = value
    return result

def max_value(values):
    """ Calculate the maximum value of a list """
    if not values:
        return 0
    result = values[0]
    for value in values:
        if value > result:
            result = value
    return result

def count(values):
    """ Calculate the count of a list """
    return len(values)

def product(values):
    """ Calculate the product of a list """
    if not values:
        return 0
    result = 1
    for value in values:
        result *= value
    return result

def cumsum(values):
    """ Calculate the cumulative sum of a list """
    return [sum_values(values[:i+1]) for i in range(len(values))]

def sqrt(x):
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