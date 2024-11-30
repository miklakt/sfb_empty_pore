#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize

P_coeffs =[]
Q_coeffs =[]

def pade_approximant_with_equation(x_vals, y_vals, m, n, constrain_Q_at = None):
    global P_coeffs, Q_coeffs
    """
    Fit a Padé approximant to the given data and return the equation as a string.

    Parameters:
        x_vals (array-like): Input x values.
        y_vals (array-like): Input y values.
        m (int): Degree of the numerator polynomial P(x).
        n (int): Degree of the denominator polynomial Q(x).

    Returns:
        tuple: A function representing the Padé approximant and its string equation.
    """
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    def rational_function(coeffs, x):
        P_coeffs = coeffs[:m + 1]
        Q_coeffs = coeffs[m + 1:] if n > 0 else [1.0]  # Default Q(x) = 1 if n = 0
        P = np.polyval(P_coeffs, x)
        Q = np.polyval(Q_coeffs, x)
        return P / Q

    def residuals(coeffs):
        r = y_vals - rational_function(coeffs, x_vals)
        return np.sum(r ** 2)

    initial_guess = np.ones(m + n + 2 if n > 0 else m + 1)
    if (constrain_Q_at is not None) and n>=1:
        def constraint_Q_at_X(coeffs):
            """
            Constraint to ensure Q(X) = 0.
            """
            Q_coeffs = coeffs[m + 1:]
            return np.polyval(Q_coeffs, constrain_Q_at)
        # Define constraints: Q(X) = 0
        constraints = {
            'type': 'eq',
            'fun': constraint_Q_at_X
        }
        result = minimize(residuals, initial_guess, constraints = constraints)
    else:
        result = minimize(
            residuals, initial_guess, 
            #method = 'SLSQP'
            )

    if not result.success:
        raise ValueError("Optimization did not converge.")

    coeffs = result.x
    P_coeffs = coeffs[:m + 1]
    Q_coeffs = coeffs[m + 1:] if n > 0 else [1.0]

    def fitted_function(x):
        return rational_function(coeffs, np.array(x))

    # Generate the string representation of P(x)
    if m == 0:
        P_str = f"{P_coeffs[0]:.5g}"
    else:
        P_terms = [f"{coeff:.5g}*x**{m - i}" if i < m else f"{coeff:.5g}" for i, coeff in enumerate(P_coeffs)]
        P_str = " + ".join(term for term in P_terms if not term.startswith("0x"))

    # Generate the string representation of Q(x)
    if n == 0:
        Q_str = "1"
    else:
        Q_terms = [f"{coeff:.5g}*x**{n - i}" if i < n else f"{coeff:.5g}" for i, coeff in enumerate(Q_coeffs)]
        Q_str = " + ".join(term for term in Q_terms if not term.startswith("0x"))

    # Final equation
    equation = f"({P_str}) / ({Q_str})"

    return fitted_function, equation

def evaluate_fit(x_vals, y_vals, fitted_function):
    y_fitted = fitted_function(x_vals)
    mse = np.mean((y_vals - y_fitted) ** 2)
    return mse

def find_best_degrees(x_vals, y_vals, max_m=5, max_n=5):
    """
    Find the best degrees (m, n) for the Padé approximant by minimizing the error.

    Parameters:
        x_vals (array-like): Input x values.
        y_vals (array-like): Input y values.
        max_m (int): Maximum degree for the numerator polynomial.
        max_n (int): Maximum degree for the denominator polynomial.

    Returns:
        tuple: Best m, best n, and the corresponding equation string.
    """
    best_m, best_n = None, None
    best_error = float('inf')
    best_equation = None

    for m in range(max_m + 1):
        for n in range(max_n + 1):
            try:
                fitted_function, equation = pade_approximant_with_equation(x_vals, y_vals, m, n)
                error = evaluate_fit(x_vals, y_vals, fitted_function)
                print(f"m={m}, n={n}, error={error}")
                if error < best_error:
                    best_error = error
                    best_m, best_n = m, n
                    best_equation = equation
                    best_func = fitted_function
            except Exception as e:
                print(f"m={m}, n={n} failed with error: {e}")

    return best_m, best_n, best_error, best_func, best_equation
#%%
relative_size = np.arange(0, 0.92, 0.02)
wall_correction_coef = np.array([
    1.00000,  1.04393,   1.09178,    1.14397,    1.20096,    1.26330,    1.33159,    1.40654,
    1.48892,   1.57966,   1.67980,    1.79054,    1.91328,    2.04963,    2.20150,    2.37109,
    2.56100,   2.77430,   3.01464,    3.28635,    3.59464,    3.94578,    4.34741,    4.80880,
    5.34141,   5.95938,   6.68043,    7.52686,    8.52705,    9.71752,    11.14580,   12.87453, 
    14.98751,  17.59862,  20.86550,   25.01092,   30.35733,   37.38467,   46.83132,   59.87967,
    78.51925,  106.31670, 150.22684,  225.51931,  372.41035,  737.25652,
    ]) #from Paine1975

best_m, best_n, best_error, approximant, equation = find_best_degrees(relative_size, (np.log(wall_correction_coef)*(1-relative_size)), max_m=2, max_n=0)
print(f"Best m={best_m}, n={best_n}, with error={best_error}")
print(equation)


#approximant = pade_approximant(relative_size, np.log(wall_correction_coef), best_m, best_n)
#%%
x = np.arange(0, 0.9, 0.005)
#y = approximant(x)/(1-x)
y = -2.18389934*(x-1.09927523)*(x+0.86222054)*x/(1-x)
#y = -2*(x-1.1)*(x+0.9)*x/(1-x)

Renkin = (1-x)**2 * (1 - 2.104*x + 2.09*x**3 - 0.95*x**5)
r=1
s=5
Renkin_pore = s/(np.pi*r**2*Renkin)
Rayleigh = 1/(2*(r-x)/(1 + 2*np.exp(y)*(s+x*2)/((r-x)*np.pi)))
Rayleigh_no_drag = 1/(2*(r-x)/(1 + 2*(s+x*2)/((r-x)*np.pi)))
# %%
#plt.scatter(relative_size, np.log(wall_correction_coef))
#plt.plot(x, y)
# y_low = 2*x
# y_high = x*0.5/(1-x)
# plt.plot(x[:], y_low)
# plt.plot(x, y_high)
plt.plot(x, Renkin_pore, label = "Renkin")
plt.plot(x, Rayleigh, label = "Rayleigh")
plt.plot(x, Rayleigh_no_drag, label = "Rayleigh no drag")
#plt.plot(x, np.exp(y2))
plt.yscale("log")
#plt.xlim(0, 0.7)
#plt.ylim(0,10000)
plt.legend()
# %%
plt.scatter(relative_size, wall_correction_coef)
#plt.plot(x, y)
# y_low = 2*x
# y_high = x*0.5/(1-x)
# plt.plot(x[:], y_low)
# plt.plot(x, y_high)
plt.plot(x, 1/Renkin, label = "Renkin")
plt.plot(x, np.exp(y), label = "Rayleigh")
#plt.plot(x, Rayleigh_no_drag, label = "Rayleigh no drag")
#plt.plot(x, np.exp(y2))
plt.yscale("log")
#plt.xlim(0, 0.7)
#plt.ylim(0,10000)
plt.legend()
# %%
