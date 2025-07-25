# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.special import lambertw
# # from scipy.optimize import minimize_scalar

# # # Parameters (make sure M_j^* > 0)
# # xi_j_plus_1 = 0.2
# # b_j = 5
# # B_j_plus_1 = 1

# # # Constants
# # a = 1 - xi_j_plus_1
# # log_a = np.log(a)
# # Upsilon = (b_j / B_j_plus_1) * log_a
# # C = Upsilon - 1

# # # Analytical optimum M_j^*
# # W_val = lambertw(-np.exp(C)).real
# # M_star_analytical = (1 - Upsilon - W_val) / log_a

# # # Define the function
# # def f(M):
# #     return (1 - a**M) / (b_j + M * B_j_plus_1)

# # # Numerical optimization using scipy
# # opt_result = minimize_scalar(lambda M: -f(M), bounds=(0.01, 20), method='bounded')
# # M_star_numerical = opt_result.x
# # f_star_numerical = -opt_result.fun

# # # Evaluate analytical result
# # f_star_analytical = f(M_star_analytical)

# # # Integer candidates
# # floor_M = int(np.floor(M_star_analytical))
# # ceil_M = int(np.ceil(M_star_analytical))
# # f_floor = f(floor_M)
# # f_ceil = f(ceil_M)

# # # Plotting range
# # M_vals = np.linspace(1, 20, 400)
# # f_vals = f(M_vals)

# # # Plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(M_vals, f_vals, label=r'$f(M_j)$', color='blue')
# # plt.axvline(M_star_analytical, color='green', linestyle='--', label=fr'Analytical $M_j^* \approx {M_star_analytical:.2f}$')
# # plt.scatter([M_star_analytical], [f_star_analytical], color='purple', s=100, label=fr'$f(M_j^*)$ (analytical)')

# # plt.axvline(M_star_numerical, color='orange', linestyle=':', label=fr'Numerical $M_j^* \approx {M_star_numerical:.2f}$')
# # plt.scatter([M_star_numerical], [f_star_numerical], color='black', s=80, marker='x', label=fr'$f(M_j^*)$ (numerical)')

# # plt.scatter([floor_M, ceil_M], [f_floor, f_ceil], color='red', zorder=5, label='Integer evaluations')
# # plt.title('Computational Decoupling Rate $f(M_j)$ with Analytical and Numerical Maxima')
# # plt.xlabel(r'$M_j$')
# # plt.ylabel(r'$f(M_j)$')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # Print results
# # print("Analytical M_j^*:", M_star_analytical)
# # print("f(M_j^*) analytical:", f_star_analytical)
# # print("Numerical M_j^*:", M_star_numerical)
# # print("f(M_j^*) numerical:", f_star_numerical)
# # print("f(floor(M_j^*)):", f_floor)
# # print("f(ceil(M_j^*)):", f_ceil)
# # print("Optimal integer M_j:", floor_M if f_floor > f_ceil else ceil_M)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize_scalar

# # Parameters (choose some positive values)
# xi = 0.3           # in (0,1)
# b = 1.0            # > 0
# B = 0.5            # > 0

# def f(M):
#     # To avoid issues at M=0, restrict domain to M > 0
#     M = np.array(M)
#     val = (1 - (1 - xi)**M) / (b + B * M)
#     return val

# # Find real maximizer M* on positive real line
# res = minimize_scalar(lambda M: -f(M), bounds=(1e-6, 20), method='bounded')
# M_star = res.x
# f_star = f(M_star)

# print(f"Real-valued maximizer M* = {M_star:.4f}, f(M*) = {f_star:.4f}")

# # Evaluate at integer neighbors
# M_floor = int(np.floor(M_star))
# M_ceil = int(np.ceil(M_star))

# # Handle edge cases if M_star is near 0 or very close to integer
# candidates = list(set([M_floor, M_ceil]))
# candidates = [m for m in candidates if m > 0]  # only positive integers

# f_candidates = [(m, f(m)) for m in candidates]
# best_int, best_val = max(f_candidates, key=lambda x: x[1])

# print(f"Integer candidates and their values:")
# for m, val in f_candidates:
#     print(f"  M = {m}, f(M) = {val:.4f}")
# print(f"Integer maximizer is M = {best_int} with f(M) = {best_val:.4f}")

# # Plotting
# M_vals = np.linspace(1e-3, 20, 500)
# f_vals = f(M_vals)

# plt.figure(figsize=(10, 6))
# plt.plot(M_vals, f_vals, label='f(M) continuous')
# plt.scatter([M_star], [f_star], color='red', label=f'Real maximizer M*={M_star:.2f}')
# plt.scatter([m for m, _ in f_candidates], [val for _, val in f_candidates], 
#             color='green', label='Integer candidates')

# plt.xlabel('M')
# plt.ylabel('f(M)')
# plt.title('Function f(M) and its maximizers')
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
from scipy.optimize import root_scalar

# Parameters
xi = 0.3
b = 1.0
B = 0.5

def f_prime(M):
    a = 1 - xi
    numerator = - (a**M) * np.log(a) * (b + B*M) - B * (1 - a**M)
    denominator = (b + B*M)**2
    return numerator / denominator

def f_double_prime(M):
    a = 1 - xi
    ln_a = np.log(a)
    numerator = (
        - (a**M) * (ln_a**2) * (b + B*M)**2
        - 2 * ( - (a**M) * ln_a ) * (b + B*M) * B
        + 2 * (1 - a**M) * B**2
    )
    denominator = (b + B*M)**3
    return numerator / denominator

sol = root_scalar(f_prime, bracket=[1e-6, 50], method='brentq')
M_star = sol.root

print(f"Numerical maximizer M* = {M_star:.6f}")
print(f"Second derivative at M*: f''(M*) = {f_double_prime(M_star):.6f}")

if f_double_prime(M_star) < 0:
    print("Function f is locally concave at M* (strict local maximum).")
else:
    print("Function f is not locally concave at M* (check calculations).")
