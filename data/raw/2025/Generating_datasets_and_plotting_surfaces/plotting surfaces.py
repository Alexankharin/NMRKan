# Go to the bottom of the code, to set up the surface you want to plot


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast




def plot_fixed_coordinates(file_path, fixed_coords, plot_type="3D", tol=1e-3):
    # Load each line and parse it
    with open(file_path, "r") as f:
        raw_lines = f.readlines()

    # Convert strings like "{1, 2, 3, 4, 5}" to lists of floats
    data = []
    for line in raw_lines:
        line = line.strip().strip("{}").replace(",", " ")
        try:
            values = list(map(float, line.split()))
            if len(values) > 1:
                data.append(values)
        except Exception:
            continue

    df = pd.DataFrame(data)
    coords = df.iloc[:, :-1]
    values = df.iloc[:, -1]

    n_coords = coords.shape[1]

    # Filter based on fixed coordinates
    for idx, val in fixed_coords.items():
        mask = np.abs(coords.iloc[:, idx - 1] - val) < tol
        coords = coords[mask]
        values = values.loc[coords.index]

    if coords.empty:
        print("❌ No matching data found for fixed coordinates.")
        return

    # Determine variable dimensions
    variable_indices = [i for i in range(n_coords) if (i + 1) not in fixed_coords]
    
    if len(variable_indices) == 1:
        # 2D plot
        x = coords.iloc[:, variable_indices[0]]
        y = values
        plt.figure()
        plt.plot(x, y, 'o-')
        plt.xlabel(f"x{variable_indices[0]+1}")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()

    elif len(variable_indices) == 2:
        # 3D plot
        x = coords.iloc[:, variable_indices[0]]
        y = coords.iloc[:, variable_indices[1]]
        z = values

        x_unique = np.sort(x.unique())
        y_unique = np.sort(y.unique())
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.full(X.shape, np.nan)

        for xi, xv in enumerate(x_unique):
            for yi, yv in enumerate(y_unique):
                mask = (np.abs(x - xv) < tol) & (np.abs(y - yv) < tol)
                if mask.any():
                    Z[yi, xi] = z[mask].values[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel(f"x{variable_indices[0]+1}")
        ax.set_ylabel(f"x{variable_indices[1]+1}")
        ax.set_zlabel("f(x)")
        plt.show()
    else:
        print("❌ Unsupported plot configuration. Must leave 1 or 2 coordinates free.")

# plot_fixed_coordinates(
# # this function plots the surfaces from the attached files. Each file contains coordinates in the first columns, and the last column contains the value of a corresponding frequency (f1, f2, or f3)
# # Set the name of the file and run python script. There are 3D datsets - e.g.Freq3_sym_v1_ks and there are 5D datasets, all of which contain "JintraDifferent" in it's title. 
# # For the 5D datasets, you need to fix certain corridantes so to have 3D or 2D cut along these coordinates. Examples are given below. Uncomment this function to run it.
#     "Freq1_sym_JintraDifferent_v0_ks.tsv",
#     fixed_coords={1: -5.0, 2: -3.0},
#     plot_type="3D"
# )

# plot_fixed_coordinates(
# # this function plots the surfaces from the attached files. Each file contains coordinates in the first columns, and the last column contains the value of a corresponding frequency (f1, f2, or f3)
# # Set the name of the file and run python script. There are 3D datsets - e.g.Freq3_sym_v1_ks and there are 5D datasets, all of which contain "JintraDifferent" in it's title. 
# # For the 5D datasets, you need to fix certain corridantes so to have 3D or 2D cut along these coordinates. examples are given below. Uncomment this function to run it.
#     "Freq1_sym_JintraDifferent_v0_ks.tsv",
#     fixed_coords={1: -5.0, 2: -3.0, 4: -1.0},
#     plot_type="2D"
# )

# plot_fixed_coordinates(
#     "Freq2_sym_v1_ks.tsv",
#     fixed_coords={},
#     plot_type="3D"
# )

plot_fixed_coordinates(
    "Freq2_sym_v1_ks.tsv",
    fixed_coords={1: -14.0},
    plot_type="2D"
)