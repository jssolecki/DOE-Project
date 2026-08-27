# # Linear Model — Configurable Order
# 
# Fits a regression model with optional term types:
# 
# | Toggle | Terms added |
# |---|---|
# | Off | `y = c + b1*x1 + b2*x2 + ...` (main effects only) |
# | Interactions | `+ bij * xi*xj` for all pairs |
# | Quadratic | `+ bii * xi²` for each input |
# 
# Inputs are normalized to [-1, 1] before fitting, then converted back to original units.

# ## Configuration — set your options here

# ## HERE IS THE CONNECTION TP THE CSV FILE *************
CSV_PATH = 'DOE Table.csv'   # <-- change this
# ## HERE IS THE CONNECTION TP THE CSV FILE *************

INCLUDE_INTERACTIONS = True    # xi * xj terms for all pairs i < j
INCLUDE_QUADRATIC    = False   # xi^2 terms for each input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

np.set_printoptions(suppress=True, precision=6)

# ## Step 1 — Load the CSV
# Last column = output (y). All other columns = inputs.

df = pd.read_csv(CSV_PATH)

print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
df.head()

# ## Step 2 — Split into inputs (X) and output (y)

output_col = df.columns[-1]
input_cols = list(df.columns[:-1])

y      = df[output_col].values
X_raw  = df[input_cols].values     # shape: (n_rows, n_inputs)

n_inputs = len(input_cols)
print(f'Output (y) : {output_col}')
print(f'Inputs ({n_inputs})  : {input_cols}')

assert 1 <= n_inputs <= 12, 'Expected between 1 and 12 input columns'

# ## Step 3 — Normalize inputs to [-1, 1]

x_min  = X_raw.min(axis=0)
x_max  = X_raw.max(axis=0)

X_norm = 2 * (X_raw - x_min) / (x_max - x_min) - 1

print('Normalization ranges:')
for i, col in enumerate(input_cols):
    print(f'  {col:25s}  [{x_min[i]:.3f}, {x_max[i]:.3f}]  ->  [-1, 1]')


# ## Step 4 — Build the model matrix
# Columns included depend on the toggles set in the config cell.

columns    = [np.ones(len(y))]   # intercept
term_names = ['const']
term_order = ['intercept']       # track order for chart coloring

# --- 1st order: main effects ---
for i, col in enumerate(input_cols):
    columns.append(X_norm[:, i])
    term_names.append(col)
    term_order.append('1st order')

# --- 2nd order: interactions (xi * xj) ---
if INCLUDE_INTERACTIONS:
    for (i, j) in combinations(range(n_inputs), 2):
        columns.append(X_norm[:, i] * X_norm[:, j])
        term_names.append(f'{input_cols[i]}*{input_cols[j]}')
        term_order.append('2nd order (interaction)')

# --- 2nd order: quadratic (xi^2) ---
if INCLUDE_QUADRATIC:
    for i, col in enumerate(input_cols):
        columns.append(X_norm[:, i] ** 2)
        term_names.append(f'{col}^2')
        term_order.append('2nd order (quadratic)')

M = np.column_stack(columns)   # shape: (n_rows, n_terms)

print(f'Model includes {len(term_names)} terms:')
for name, order in zip(term_names, term_order):
    print(f'  {name:30s}  [{order}]')

# ## Step 5 — Solve OLS via Normal Equations
# **b = (MᵀM)⁻¹ Mᵀy** — coefficients in normalized space.

MT_M   = M.T @ M
MT_y   = M.T @ y
b_norm = np.linalg.solve(MT_M, MT_y)

print('Coefficients (normalized input space):')
for name, coef in zip(term_names, b_norm):
    print(f'  {name:30s}  {coef:.6f}')

# ## Step 6 — Convert coefficients back to original units
# Each normalized input satisfies `x_norm_i = a_i * x_i + offset_i`,
# so substituting back distributes each coefficient across the intercept and original-scale terms.

a      = 2.0 / (x_max - x_min)               # scale factor per input
offset = -(x_max + x_min) / (x_max - x_min)  # offset per input

b_orig = np.zeros(len(term_names))

for t_idx, name in enumerate(term_names):
    b_t = b_norm[t_idx]

    if name == 'const':
        b_orig[0] += b_t

    elif '^2' in name:
        # Quadratic: x_norm_i^2 = (a_i*x_i + off_i)^2
        #          = a_i^2 * x_i^2  +  2*a_i*off_i * x_i  +  off_i^2
        base = name.replace('^2', '')
        i    = input_cols.index(base)
        i_main = 1 + i                              # index of x_i main effect

        b_orig[0]       += b_t * offset[i] ** 2    # -> intercept
        b_orig[i_main]  += b_t * 2 * a[i] * offset[i]  # -> x_i
        b_orig[t_idx]   += b_t * a[i] ** 2         # -> x_i^2

    elif '*' in name:
        # Interaction: x_norm_i * x_norm_j = (a_i*x_i + off_i)(a_j*x_j + off_j)
        parts  = name.split('*')
        i, j   = input_cols.index(parts[0]), input_cols.index(parts[1])
        i_main = 1 + i
        j_main = 1 + j

        b_orig[0]       += b_t * offset[i] * offset[j]   # -> intercept
        b_orig[i_main]  += b_t * a[i] * offset[j]        # -> x_i
        b_orig[j_main]  += b_t * a[j] * offset[i]        # -> x_j
        b_orig[t_idx]   += b_t * a[i] * a[j]             # -> x_i*x_j

    else:
        # Main effect: x_norm_i = a_i * x_i + off_i
        i = input_cols.index(name)
        b_orig[0]      += b_t * offset[i]   # -> intercept
        b_orig[t_idx]  += b_t * a[i]        # -> x_i

print('Coefficients (original input units):')
for name, coef in zip(term_names, b_orig):
    print(f'  {name:30s}  {coef:.6f}')

# ## Step 7 — Print the fitted equation in original units

equation_terms = [f'{b_orig[0]:.4f}']

for name, coef in zip(term_names[1:], b_orig[1:]):
    sign = '+' if coef >= 0 else '-'
    equation_terms.append(f'{sign} {abs(coef):.4f}*{name}')

print(f'y = {"  ".join(equation_terms)}')

# ## Step 8 — Model fit statistics

y_hat     = M @ b_norm
residuals = y - y_hat

SS_res = np.sum(residuals ** 2)
SS_tot = np.sum((y - np.mean(y)) ** 2)

n_obs   = len(y)
n_terms = len(term_names)

R2      = 1 - SS_res / SS_tot
R2_adj  = 1 - (SS_res / (n_obs - n_terms)) / (SS_tot / (n_obs - 1))

print('=== Model Fit Statistics ===')
print(f'  Interactions included : {INCLUDE_INTERACTIONS}')
print(f'  Quadratic included    : {INCLUDE_QUADRATIC}')
print(f'  R²                    : {R2:.4f}   ({R2*100:.1f}% of variance explained)')
print(f'  Adjusted R²           : {R2_adj:.4f}')
print(f'  Observations          : {n_obs}')
print(f'  Terms in model        : {n_terms}  (including intercept)')

# ## Step 9 — Coefficient arrays

coef_vector_normalized = b_norm.copy()
coef_vector_original   = b_orig.copy()

print('Coefficient vector (normalized space):')
print(coef_vector_normalized)
print()
print('Coefficient vector (original units):')
print(coef_vector_original)

# ## Step 10 — Coefficient table

# ## Step 11 — Bar chart: coefficients by order
# Each order gets its own panel, sorted by magnitude so the most influential terms are easiest to spot.
# Blue = positive effect, Red = negative effect. Uses normalized coefficients.

#Convert coefficients (normalized space) to DOE main effects to DOR 2-factor interactions (JSS)

b_norm = 2*b_norm

# Group terms by order, skipping the intercept
order_groups = {}
for name, coef, order in zip(term_names, b_norm, term_order):
    if order == 'intercept':
        continue
    order_groups.setdefault(order, []).append((name, coef))

n_panels = len(order_groups)

if n_panels == 0:
    print('No terms to plot.')
else:
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(4, len(term_names) * 0.4)))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle('Coefficient Estimates by Order (normalized input space)', fontsize=14)

    def plot_coef_bars(ax, names, coefs, title):
        # Sort by magnitude, largest at top
        sorted_pairs = sorted(zip(coefs, names), key=lambda x: abs(x[0]))
        coefs_sorted  = [p[0] for p in sorted_pairs]
        names_sorted  = [p[1] for p in sorted_pairs]

        colors = ['steelblue' if c >= 0 else 'tomato' for c in coefs_sorted]

        ax.barh(names_sorted, coefs_sorted, color=colors, edgecolor='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Coefficient value')

        # Label each bar with its value
        max_abs = max(abs(c) for c in coefs_sorted) if coefs_sorted else 1
        for i, coef in enumerate(coefs_sorted):
            h_offset = 0.02 * max_abs
            ax.text(coef + h_offset if coef >= 0 else coef - h_offset, i,
                    f'{coef:.3f}', va='center',
                    ha='left' if coef >= 0 else 'right', fontsize=9)

    for ax, (order_label, pairs) in zip(axes, order_groups.items()):
        names, coefs = zip(*pairs)
        plot_coef_bars(ax, names, coefs, order_label.title())

    plt.tight_layout()
    plt.show()

