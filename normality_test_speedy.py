import os
import sys
import pickle
import numpy as np
from scipy.stats import shapiro
from netCDF4 import Dataset

def load_data(file_path, variable_name):
    with Dataset(file_path, 'r') as nc:
        data = nc.variables[variable_name][:]
        sigma_levels = nc.variables['lev'][:]
    
    return data, sigma_levels

def compute_shapiro_wilk(data):
    p_values = np.empty(data.shape[1:])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            sample = data[:, i, j]
            p_values[i, j] = shapiro(sample)[1]
    return p_values

def calculate_theoretical_pressure(sigma):
    return sigma * 1000

def main():
    # Parse command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python normality_test_speedy.py <days_since_20110101> <ensemble_name> <variable_name> <output_dir>")
        sys.exit(1)

    days_since_20110101 = int(sys.argv[1])
    ensemble_name = sys.argv[2]
    variable_name = sys.argv[3]
    output_dir = sys.argv[4]

    # Path to the SPEEDY data directory
    base_path = "/fs/ess/PAS2856/SPEEDY_ensemble_data"
    ensemble_path = os.path.join(base_path, ensemble_name)

    # Generate file name based on days since Jan 1, 2011
    from datetime import datetime, timedelta
    start_date = datetime(2011, 1, 1)
    target_date = start_date + timedelta(days=days_since_20110101)
    file_name = target_date.strftime("%Y%m%d%H00") + ".nc"
    file_path = os.path.join(ensemble_path, file_name)

    # Debugging: Print file path
    print(f"Attempting to load file: {file_path}")

    # Load data
    print(f"Loading data from {file_path}...")
    data, sigma = load_data(file_path, variable_name)

    # Compute Shapiro-Wilk p-values
    print("Computing Shapiro-Wilk test p-values...")
    p_values = compute_shapiro_wilk(data)

    # Compute theoretical pressure levels
    print("Calculating theoretical pressure levels...")
    theoretical_pressure = calculate_theoretical_pressure(sigma)

    # Save results to a pickle file
    output_file = f"{variable_name}_{ensemble_name}_{target_date.strftime('%Y%m%d%H%M')}_pvalues.pkl"
    output_path = os.path.join(output_dir, output_file)
    results = {
        "date": target_date.strftime("%Y%m%d%H%M"),
        "vname": variable_name,
        "pvalues": p_values,
        "theoretical_pressure": theoretical_pressure
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
# bash command is python normality_test_speedy.py 0 reference_ens u ./output
# Hello