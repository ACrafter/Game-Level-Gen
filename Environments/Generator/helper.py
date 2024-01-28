import random

import numpy as np
import sympy

MAX = 100


def gen_level(size, seed):
    total_number_of_tiles = size * size
    if seed is not None:
        np.random.seed(seed)

    board = np.random.choice(np.arange(MAX + 1), total_number_of_tiles, replace=True)
    np.random.shuffle(board)
    board = np.reshape(board, (size, size))
    return board


def calculate_rewards_PCGRL(current, old, ideal, accepted_variation, max_only=False):
    if not max_only:
        low = ideal - accepted_variation
    else:
        low = 1

    high = ideal + accepted_variation

    if low <= current <= high and low <= old <= high:
        return 0
    if old <= high and current <= high:
        return min(current, low) - min(old, low)
    if old >= low and current >= low:
        return max(old, high) - max(current, high)
    if current > high and old < low:
        return high - current + old - low
    if current < low and old > high:
        return high - old + current - low


def check_primes(level):
    primes = []
    for e in level.flatten():
        if sympy.isprime(e):
            primes.append(e)

    return primes, len(primes)


def calculate_rewards_with_in_range_method(current, ideal, accepted_variation, weight):
    low = ideal - accepted_variation
    high = ideal + accepted_variation

    if low <= current <= high:
        return weight
    else:
        return -weight * 10


# print(calculate_rewards_with_in_range_method(8, 3, 2, 0.1))
