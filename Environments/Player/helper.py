import numpy as np
import sympy


def gen_board(size, current_max, number_of_primes):
    number_of_primes = int(number_of_primes)
    prime_numbers = list(sympy.primerange(1, current_max + 1))

    possible_non_primes = np.arange(1, current_max)
    non_prime_numbers = possible_non_primes[~np.isin(possible_non_primes, prime_numbers)]
    prime_cells = np.random.choice(prime_numbers, number_of_primes, replace=True)

    number_of_non_primes = (size * size) - number_of_primes
    non_prime_cells = np.random.choice(non_prime_numbers, number_of_non_primes, replace=True)

    board = np.append(prime_cells, non_prime_cells)
    np.random.shuffle(board)
    board = np.reshape(board, (size, size))

    return board, prime_cells
