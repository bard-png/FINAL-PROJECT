import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control
from netCDF4 import Dataset


def load_pvalues(start_date, end_date, ensemble_name, variable_name, day_interval, data_dir):
    """
    Load p-values from pickle files for the specified variable, ensemble, and date range.
    It returns a combined 4D array of p-values (time, lev, lat, lon).
    """
    dates = np.arange(np.datetime64(start_date), np.datetime64(end_date), np.timedelta64(day_interval, 'D'))
    combined_pvals = []
    days = 150
    day_count = 0

    for date in dates:
        if day_count >= days:
            print(f"Stopping after {days} days.")
            break
        
        date_str = date.astype(str).replace('-', '')
        filename = f"{data_dir}/{variable_name}_{ensemble_name}_{date_str}0000_pvalues.pkl"
        
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            combined_pvals.append(data["pvalues"])
        
        day_count += 1
    
    if not combined_pvals:
        print("No valid pickle files loaded.")
        sys.exit(1)

    # Create 4D array (time, lev, lat, lon)
    return np.stack(combined_pvals, axis=0)


def perform_fdr_adjustment(pvals):
    """
    Perform FDR adjustment using the Benjamini-Yekutieli procedure.
    """
    shape = pvals.shape
    flattened_pvals = pvals.flatten()
    
    adjusted_pvals = false_discovery_control(flattened_pvals, method="by")
    
    return adjusted_pvals.reshape(shape)


def plot_rejections(rejections, title, output_filename):
    """
    Plot the null hypothesis rejections as a function of latitude, model level, or time.
    """
    swapped_rejections = np.sum(rejections, axis=(1, 2)).T

    plt.figure(figsize=(10, 6))
    plt.imshow(swapped_rejections, aspect='auto', cmap='Reds')
    plt.colorbar(label='Number of Null Hypothesis Rejections')
    plt.title(title)
    plt.xlabel('Time (Index)')
    plt.ylabel('Model Levels')
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()

def main():
    # Parse command-line inputs
    if len(sys.argv) != 7:
        print("Usage: python examine_normality_test_pvals.py <start_date> <end_date> <ensemble_name> <variable_name> <day_interval> <data_dir>")
        sys.exit(1)
    
    start_date = sys.argv[1]  # example, "2011-06-01"
    end_date = sys.argv[2]    # example, "2011-06-10"
    ensemble_name = sys.argv[3]  # example, "reference_ens"
    variable_name = sys.argv[4]  # example, "u"
    day_interval = int(sys.argv[5])  # example, 1
    data_dir = sys.argv[6]  # Directory containing pickle files
    
    # Load p-values
    print("Loading p-values from pickle files")
    combined_pvals = load_pvalues(start_date, end_date, ensemble_name, variable_name, day_interval, data_dir)
    print("P-values loaded.")
    
    # Perform FDR adjustment
    print("Performing FDR adjustment")
    adjusted_pvals = perform_fdr_adjustment(combined_pvals)
    rejections = adjusted_pvals < 0.05  # Null hypothesis rejections
    print("FDR adjustment complete.")
    
    # Plot rejections
    title = f"Null Hypothesis Rejections ({variable_name}, {ensemble_name})"
    output_filename = f"{variable_name}_{ensemble_name}_rejections.png"
    plot_rejections(rejections, title, output_filename)


if __name__ == "__main__":
    main()
