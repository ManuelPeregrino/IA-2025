import math

def calculate_n(a, b, delta_x):
    """
    Calculate the number of steps (n) between a range [A, B] with a given resolution \( \Delta x \).
    """
    n = math.ceil((b - a) / delta_x) + 1
    return n

def calculate_bits_required(n):
    """
    Calculate the number of bits required to represent n points.
    """
    bits = math.ceil(math.log2(n))
    return bits

def calculate_system_resolution(a, b, bits):
    """
    Calculate the real resolution (\( \Delta x^* \)) achievable with the given number of bits.
    """
    levels = 2 ** bits - 1
    delta_x_star = (b - a) / levels
    return delta_x_star

def calculate_point(a, i, delta_x_star):
    """
    Calculate the i-th point (X) in the range using \( X = A + i \cdot \Delta x^* \).
    """
    x = a + i * delta_x_star
    return x