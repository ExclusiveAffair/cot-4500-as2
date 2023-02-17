import numpy as np

def print_double_spaced(x):
    print(x)
    print()

# returns the (n-1)th degree interpolating polynomial value at x for the (x, y) points given
def nevilles_method(xs, ys, x):
    n = len(xs)
    neville_matrix = np.zeros((n, n))
    for i in range(n):
        neville_matrix[i][0] = ys[i]
    for i in range(1, n):
        for j in range(1, i+1):
            term1 = (x - xs[i-j]) * neville_matrix[i][j-1]
            term2 = (x - xs[i]) * neville_matrix[i-1][j-1]
            neville_matrix[i][j] = (term1 - term2) / (xs[i] - xs[i-j])
    return neville_matrix[n-1][n-1]

# returns the divided differences given a series of (x, y) pairs
def gen_divided_differences(xs, ys):
    n = len(xs)
    differences_matrix = np.zeros((n, n+1))
    for i in range(n):
        differences_matrix[i][n] = xs[i]
        differences_matrix[i][0] = ys[i]

    for i in range(1, n):
        for j in range(1, i+1):
            differences_matrix[i][j] = (differences_matrix[i][j-1] - differences_matrix[i-1][j-1]) / (xs[i] - xs[i-j])
    
    differences = []
    for i in range(1, n): differences.append(differences_matrix[i][i])
    print_double_spaced(differences)
    differences.insert(0, differences_matrix[0][0])
    return differences

# returns the divided differences for Hermite polynomial interpolation given a series of (x, y) pairs
def gen_hermite_divided_differences(xs, ys, derivs):
    n = len(xs)
    differences_matrix = np.zeros((2 * n, 2 * n))

    for i in range(2 * n):
        differences_matrix[i][0] = xs[i//2]
        differences_matrix[i][1] = ys[i//2]

    for i in range(1, 2 * n, 2):
        differences_matrix[i][2] = derivs[(i-1)//2]
    
    for i in range(1, 2 * n):
        for j in range(2, i+2):
            if j == 2 and i % 2 == 1: continue
            if j >= 2 * n: continue
            differences_matrix[i][j] = (differences_matrix[i][j-1] - differences_matrix[i-1][j-1]) / (xs[i//2] - xs[(i-j+1)//2])

    return differences_matrix

# returns the approximation for f(x) given the divided differences matrix and the desired degree
def val_from_divided_differences_poly(differences, degree, x, xs):
    y_val = 0
    span_product = 1
    for i in range(0, degree + 1):
        y_val += differences[i] * span_product
        span_product *= x - xs[i]
    return y_val

# returns matrix A for cubic spline interpolation
def gen_matrix_a(xs, ys, n, h):
    a = np.zeros((n+1, n+1))
    a[0][0] = a[n][n] = 1

    for i in range(1, n):
        a[i][i] = 2 * (h[i-1] + h[i])

    for i in range(1, n):
        a[i][i+1] = h[i]
        a[i][i-1] = h[i-1]
    
    return a

# returns matrix B for cubic spline interpolation
def gen_matrix_b(xs, ys, n, h):
    b = np.zeros(n+1)

    for i in range(1, n):
        b[i] = 3 / h[i] * (ys[i+1] - ys[i]) - 3 / h[i-1] * (ys[i] - ys[i-1])

    return b

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    print_double_spaced(nevilles_method([3.6, 3.8, 3.9], [1.675, 1.436, 1.318], 3.7))

    differences = gen_divided_differences([7.2, 7.4, 7.5, 7.6], [23.5492, 25.3913, 26.8224, 27.4589])
    print_double_spaced(val_from_divided_differences_poly(differences, 3, 7.3, [7.2, 7.4, 7.5, 7.6]))

    print_double_spaced(gen_hermite_divided_differences([3.6, 3.8, 3.9], [1.675, 1.436, 1.318], [-1.195, -1.188, -1.182]))

    cubic_spline_x = [2, 5, 8, 10]
    cubic_spline_y = [3, 5, 7, 9]
    
    n = len(cubic_spline_x) - 1
    h = []
    for i in range(1, n+1):
        h.append(cubic_spline_x[i] - cubic_spline_x[i-1])

    a = gen_matrix_a(cubic_spline_x, cubic_spline_y, n, h)
    b = gen_matrix_b(cubic_spline_x, cubic_spline_y, n, h)

    print_double_spaced(a)
    print_double_spaced(b)
    print_double_spaced(np.linalg.solve(a, b))