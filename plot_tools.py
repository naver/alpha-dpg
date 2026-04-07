# alpha-dpg
# Copyright (C) 2026 Naver Corporation. All rights reserved.

import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re
from matplotlib.collections import PathCollection
from adjustText import adjust_text

def combine_path_collections(
    pathcollections: list[PathCollection],
) -> PathCollection:
    """
    Correctly combines multiple PathCollection objects into a single one.

    This function gathers not only paths and sizes but also the crucial
    offsets (coordinates) and colors.
    """
    # Initialize lists to store all properties
    all_paths = []
    all_sizes = []
    all_offsets = []
    all_facecolors = []
    all_edgecolors = []

    # Loop through each collection and gather its properties
    for pc in pathcollections:
        # It's possible a collection is empty, so we check
        if pc.get_paths():
            all_paths.extend(pc.get_paths() * len(pc.get_offsets()))
            all_sizes.extend(pc.get_sizes())
            all_offsets.extend(pc.get_offsets())
            all_facecolors.extend(pc.get_facecolors())
            all_edgecolors.extend(pc.get_edgecolors())

    # Create the new collection with the visual properties
    combined_pc = PathCollection(
        all_paths,
        sizes=all_sizes,
        facecolors=all_facecolors,
        edgecolors=all_edgecolors,
    )

    # **Crucially, set the offsets on the new collection**
    combined_pc.set_offsets(np.array(all_offsets))

    return combined_pc


def pass_at_k(n, c, k):
    """
    Calculates the pass@k metric.
    
    Args:
        n (int): Total number of samples.
        c (int): Number of correct samples.
        k (int): The 'k' in pass@k.
        
    Returns:
        float: The pass@k score.
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # Handle potential overflow with large numbers by calculating log combinations
    if n > 1000: # Heuristic threshold
        try:
            from scipy.special import comb
            return 1.0 - comb(n - c, k, exact=False) / comb(n, k, exact=False)
        except ImportError:
            pass # Fallback to math.comb if scipy is not available
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_pass_at_k_std(results_series, k, n_resamples=100):
    """
    Calculates the standard deviation (standard error) for pass@k 
    using bootstrap resampling.
    
    Args:
        results_series (pd.Series): A series where each element is a list of results (0s and 1s).
        k (int): The 'k' in pass@k.
        n_resamples (int): The number of bootstrap resamples to perform.
        
    Returns:
        float: The standard deviation of the pass@k metric.
    """
    bootstrap_stats = []
    for _ in range(n_resamples):
        boot_sample = results_series.sample(n=len(results_series), replace=True)
        pass_k_stat = boot_sample.apply(lambda x: pass_at_k(len(x), int(sum(x)), k)).mean()
        bootstrap_stats.append(pass_k_stat)
    return np.std(bootstrap_stats)

# def bootstrap_pass_at_k_std_from_accuracies(accuracies, k, n_samples_json, n_resamples=1000):
#     """
#     Calculates the standard deviation (standard error) for pass@k 
#     using bootstrap resampling from a list of accuracies.
    
#     Args:
#         accuracies (list of float): A list of per-problem accuracies.
#         k (int): The 'k' in pass@k.
#         n_samples_json (int): The constant number of samples to assume per problem.
#         n_resamples (int): The number of bootstrap resamples to perform.
        
#     Returns:
#         float: The standard deviation of the pass@k metric.
#     """
#     bootstrap_stats = []
#     accuracies_np = np.array(accuracies)
#     for _ in range(n_resamples):
#         boot_sample = np.random.choice(accuracies_np, size=len(accuracies_np), replace=True)
#         pass_k_vals = [pass_at_k(n_samples_json, int(round(acc * n_samples_json)), k) for acc in boot_sample]
#         bootstrap_stats.append(np.mean(pass_k_vals))
#     return np.std(bootstrap_stats)

def catmull_rom_spline(P, n_points=100):
    """
    Compute Catmull-Rom spline for a list of points P (shape (m,2)).
    This implementation duplicates endpoints to make the spline pass through all original points.
    
    Args:
        P (np.array): Array of control points.
        n_points (int): Number of points to generate per segment.
        
    Returns:
        np.array: Array of points on the spline.
    """
    P = np.asarray(P)
    if len(P) < 2:
        return P.copy()
    pts = np.vstack([P[0], P, P[-1]])
    spline_pts = []
    for i in range(len(P)-1):
        p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]
        for t in np.linspace(0, 1, n_points, endpoint=False):
            t2, t3 = t * t, t * t * t
            f1 = -0.5*t3 + t2 - 0.5*t
            f2 =  1.5*t3 - 2.5*t2 + 1.0
            f3 = -1.5*t3 + 2.0*t2 + 0.5*t
            f4 =  0.5*t3 - 0.5*t2
            point = f1*p0 + f2*p1 + f3*p2 + f4*p3
            spline_pts.append(point)
    spline_pts.append(P[-1])
    return np.array(spline_pts)

def plot_spline_pareto_comparison(models, model_name_map, paths, k_values=(1, 256), n_samples_json=128, use_bootstrap=True):
    """
    Generates a plot with a Catmull-Rom spline Pareto front, with a fallback data loading strategy.
    
    Args:
        models (list): List of model directory names.
        model_name_map (dict): Dictionary to map model names to display names.
        paths (list or str): List of base paths to search for model data.
        k_values (tuple): A tuple of two integers (k1, k2) for the x and y axes.
        n_samples_json (int): Constant number of samples to assume for JSON data.
        use_bootstrap (bool): If True, computes and plots bootstrap standard error.
    """
    if not isinstance(paths, list):
        paths = [paths]

    plt.figure(figsize=(12, 10))
    k1, k2 = k_values
    
    plot_data = {}

    print("Searching for result files and calculating pass@k...")
    for model in models:
        mean_pass_k1, mean_pass_k2 = None, None
        std_pass_k1, std_pass_k2 = 0.0, 0.0 # Default std to 0
        found_data = False

        for path in paths:
            if found_data:
                break
            # --- STRATEGY 1: Look for Parquet file ---
            for parquet_file in [os.path.join(path, model, "results.parquet"), os.path.join(path, model, "merged_results.parquet"), os.path.join(path, model + '.eval.parquet')]:
                if os.path.exists(parquet_file):
                    print(f"  - Found parquet for '{model}' in '{path}'")
                    ds = pd.read_parquet(parquet_file)
                    results_series = ds['results']
                    print("num samples:", len(results_series[0]))
                    
                    mean_pass_k1 = results_series.apply(lambda x: pass_at_k(len(x), int(sum(x)), k1)).mean()
                    mean_pass_k2 = results_series.apply(lambda x: pass_at_k(len(x), int(sum(x)), k2)).mean()
                    if use_bootstrap:
                        print(f"    - Calculating bootstrap std error for {model} from Parquet...")
                        std_pass_k1 = bootstrap_pass_at_k_std(results_series, k=k1)
                        std_pass_k2 = bootstrap_pass_at_k_std(results_series, k=k2)

                    found_data = True
                    break

            # --- STRATEGY 2: Fallback to JSON file ---
            json_file = os.path.join(path, f"{model}.eval.json")
            if os.path.exists(json_file):
                print(f"  - Found JSON for '{model}' in '{path}' (fallback)")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if not data or not isinstance(list(data.values())[0], list):
                    print(f"    Warning: JSON file for {model} has an unexpected format. Skipping.")
                    continue
                
                accuracies = list(data.values())[0]
                pass_k1_vals = [pass_at_k(n_samples_json, int(round(acc * n_samples_json)), k1) for acc in accuracies]
                pass_k2_vals = [pass_at_k(n_samples_json, int(round(acc * n_samples_json)), k2) for acc in accuracies]
                mean_pass_k1 = np.mean(pass_k1_vals)
                mean_pass_k2 = np.mean(pass_k2_vals)
                
                if use_bootstrap:
                    print(f"    - Calculating bootstrap std error for {model} from JSON...")
                    std_pass_k1 = bootstrap_pass_at_k_std_from_accuracies(accuracies, k1, n_samples_json)
                    std_pass_k2 = bootstrap_pass_at_k_std_from_accuracies(accuracies, k2, n_samples_json)
                
                found_data = True
                break

        if not found_data:
            print(f"Warning: No results file found for model '{model}' in any provided path. Skipping.")
            continue
            
        label = model_name_map.get(model, model)
        plot_data[model] = {
            'label': label,
            'pass_k1': mean_pass_k1, 'pass_k2': mean_pass_k2,
            'std_k1': std_pass_k1, 'std_k2': std_pass_k2
        }
        print(f"    - Model: {label}")
        print(f"      pass@{k1}:   {mean_pass_k1:.4f} (± {std_pass_k1:.4f})")
        print(f"      pass@{k2}: {mean_pass_k2:.4f} (± {std_pass_k2:.4f})")

    # --- Plotting Logic ---
    if not plot_data:
        print("\nNo data was found to plot. Exiting.")
        return
        
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']
    texts = []
    text_offset = 0#-0.005
    xs, ys = [], []
    points = []
    for i, (model, data) in enumerate(plot_data.items()):
        xs.append(data['pass_k1'])
        ys.append(data['pass_k2'])
        if use_bootstrap:
            plt.errorbar(data['pass_k1'], data['pass_k2'], yerr=data['std_k2'], xerr=data['std_k1'],
                         #fmt=markers[i % len(markers)], markersize=8, capsize=4, alpha=0.9,
                         fmt=markers['DPG' in data['label']], markersize=8, capsize=4, alpha=0.9,
                         zorder=3, label=data['label']).lines
        else:
            points.append(plt.scatter(data['pass_k1'], data['pass_k2'], s=100, marker=markers[int('DPG' in data['label'])], #marker=markers[i % len(markers)],
                        c='C0' if 'DPG' in data['label'] else 'C1',
                        alpha=0.9, zorder=3, label=data['label']))

        texts.append(plt.text(data['pass_k1'], data['pass_k2']+text_offset, data['label'], fontsize=15,
                              #bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3'),
                              ha='center', va='top',
                              zorder=4))

    # --- Automatic Pareto Front Calculation ---
    if len(plot_data) >= 2:
        all_points = np.array([(data['pass_k1'], data['pass_k2']) for data in plot_data.values()])
        
        # Sort points by x-value (pass@k1) in descending order to find the front
        sorted_indices = np.argsort(all_points[:, 0])[::-1]
        sorted_points = all_points[sorted_indices]
        
        pareto_front_points = []
        max_y = -1.0
        
        for point in sorted_points:
            # A point is on the front if its y-value is greater than the max y-value seen so far
            if point[1] > max_y:
                pareto_front_points.append(point)
                max_y = point[1]
        
        pareto_front_points = np.array(pareto_front_points)
        # Re-sort by x-axis for correct spline plotting
        pareto_front_points = pareto_front_points[np.argsort(pareto_front_points[:, 0])]
        
        if len(pareto_front_points) >= 2:
            spline_pts = catmull_rom_spline(pareto_front_points, n_points=200)
            plt.plot(spline_pts[:,0], spline_pts[:,1], linestyle=':', linewidth=2.2, label='Pareto Frontier', zorder=2)
            print(f"\nPareto spline built through {len(pareto_front_points)} automatically identified points.")
        else:
            print("\nWarning: Not enough points found for the Pareto front (need at least 2).")

    if points:
       points = combine_path_collections(points)
    adjust_text(texts, objects=points if points else None, expand=(1.5,2.5),  target_x=xs, target_y=ys, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    # --- Final Plot Styling and Saving ---
    plt.xlabel(f"Precision (pass@{k1})", fontsize=16)
    plt.ylabel(f"Coverage (pass@{k2})", fontsize=16)
    title_suffix = "(with 1 Std. Error)" if use_bootstrap else ""
    #plt.title(f"Model Performance: Coverage (pass@{k2}) vs Precision (pass@{k1}) {title_suffix}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    # --- Sort Legend ---
    handles, labels = plt.gca().get_legend_handles_labels()
    # Separate Pareto front from other labels to place it last
    pareto_handle, pareto_label = None, None
    model_items = []
    for handle, label in zip(handles, labels):
        if label == 'Pareto Frontier':
            pareto_handle, pareto_label = handle, label
        else:
            model_items.append((label, handle))
    
    # Sort model labels alphabetically
    model_items.sort(key=lambda x: x[0])
    
    # Rebuild sorted handles and labels
    sorted_labels = [item[0] for item in model_items]
    sorted_handles = [item[1] for item in model_items]
    
    # Add Pareto front at the end if it exists
    if pareto_handle:
        sorted_labels.append(pareto_label)
        sorted_handles.append(pareto_handle)

    #plt.legend(sorted_handles, sorted_labels, loc='best', fontsize=13)
    plt.legend([pareto_handle], [pareto_label], loc='best', fontsize=15)
    plt.tight_layout()

    figure_dir = "./figures"
    os.makedirs(figure_dir, exist_ok=True)
    save_path = os.path.join(figure_dir, f"math_spline_pareto_front_bootstrap_{use_bootstrap}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)

    print(f"\nPlot saved to {save_path}")
    print(f"Displaying Pass@{k2} vs Pass@{k1} plot...")
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    import pandas as pd
import math
import os
import numpy as np
import json
from scipy.special import comb

# --- Helper Functions (Same as before) ---

def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    if n > 1000: 
        try:
            return 1.0 - comb(n - c, k, exact=False) / comb(n, k, exact=False)
        except ImportError:
            pass 
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def load_model_data(model_name, paths, n_samples_json=128):
    for path in paths:
        # Check Parquet
        for parquet_file in [os.path.join(path, model_name, "results.parquet"), 
                             os.path.join(path, model_name, "merged_results.parquet"), 
                             os.path.join(path, model_name + '.eval.parquet')]:
            if os.path.exists(parquet_file):
                ds = pd.read_parquet(parquet_file)
                return 'parquet', ds['results']

        # Check JSON
        json_file = os.path.join(path, f"{model_name}.eval.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            if data and isinstance(list(data.values())[0], list):
                return 'json', np.array(list(data.values())[0])
    
    print(f"Warning: Data not found for {model_name}")
    return None, None

def get_bootstrap_distribution(data_type, data, k, bootstrap_indices, n_samples_json=128):
    if data_type == 'parquet':
        data_arr = data.values
        per_problem_scores = np.array([pass_at_k(len(x), int(sum(x)), k) for x in data_arr])
        resampled_scores = per_problem_scores[bootstrap_indices]
        return np.mean(resampled_scores, axis=1)
    elif data_type == 'json':
        per_problem_scores = np.array([
            pass_at_k(n_samples_json, int(round(acc * n_samples_json)), k) 
            for acc in data
        ])
        resampled_scores = per_problem_scores[bootstrap_indices]
        return np.mean(resampled_scores, axis=1)
    return np.array([])

def calculate_p_value(dist_a, dist_b):
    diffs = dist_a - dist_b
    observed_diff = np.mean(diffs)
    if observed_diff > 0:
        p_val = 2 * np.mean(diffs <= 0)
    else:
        p_val = 2 * np.mean(diffs >= 0)
    return max(0.0, min(1.0, p_val)), observed_diff

def escape_latex(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("α", "$\\alpha$")

# --- Matrix Generation Logic ---

def print_latex_matrix(models, model_map, distributions, k_list):
    """
    Generates a LaTeX matrix where:
    - Diagonal: Absolute Score (Pass@1 / Pass@256)
    - Upper Triangle: Pass@256 Difference (Row - Col)
    - Lower Triangle: Pass@1 Difference (Row - Col)
    """
    n = len(models)
    k1, k2 = k_list[0], k_list[1] # k1=1, k2=256
    
    print("\n" + "="*20 + " LaTeX Matrix Table " + "="*20)
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\scriptsize") # Small font to fit many columns
    print(r"\setlength{\tabcolsep}{3pt}") # Tight columns
    
    # Dynamic column definition
    print(r"\begin{tabular}{l" + "c" * n + "}")
    print(r"\toprule")
    
    # --- Header Row (Vertical Names) ---
    header = " & "
    for m in models:
        display_name = model_map.get(m, m)
        header += f"\\rotatebox{{90}}{{{escape_latex(display_name)}}} & "
    header = header[:-2] + r"\\"
    print(header)
    print(r"\midrule")
    
    # --- Matrix Rows ---
    for i, row_model in enumerate(models):
        row_label = model_map.get(row_model, row_model)
        row_str = f"{escape_latex(row_label)} & "
        
        for j, col_model in enumerate(models):
            
            # -- Diagonal: Absolute Scores --
            if i == j:
                abs_k1 = np.mean(distributions[(row_model, k1)]) * 100
                abs_k2 = np.mean(distributions[(row_model, k2)]) * 100
                # Format: 12.5 / 45.2
                row_str += f"\\textbf{{{abs_k1:.1f}/{abs_k2:.1f}}} & "
            
            # -- Upper Triangle: Pass@256 (k2) Differences --
            elif j > i:
                # Calculate Row - Col
                dist_row = distributions[(row_model, k2)]
                dist_col = distributions[(col_model, k2)]
                p_val, diff = calculate_p_value(dist_row, dist_col)
                
                diff_pct = diff * 100
                cell_text = f"{diff_pct:+.1f}"
                
                # Bold if significant
                if p_val < 0.05:
                    cell_text = f"\\textbf{{{cell_text}}}"
                
                # Optional: Gray out insignificant results for clarity
                # else: cell_text = f"\\textcolor{{gray}}{{{cell_text}}}"

                row_str += f"{cell_text} & "

            # -- Lower Triangle: Pass@1 (k1) Differences --
            elif j < i:
                # Calculate Row - Col
                dist_row = distributions[(row_model, k1)]
                dist_col = distributions[(col_model, k1)]
                p_val, diff = calculate_p_value(dist_row, dist_col)
                
                diff_pct = diff * 100
                cell_text = f"{diff_pct:+.1f}"
                
                if p_val < 0.05:
                    cell_text = f"\\textbf{{{cell_text}}}"
                
                row_str += f"{cell_text} & "
        
        # Cleanup row end
        row_str = row_str[:-2] + r"\\"
        print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Pairwise performance comparison. \textbf{Diagonal}: Absolute Pass@1 / Pass@256 scores (\%). \textbf{Upper Triangle}: Pass@256 differences (Row $-$ Column). \textbf{Lower Triangle}: Pass@1 differences (Row $-$ Column). \textbf{Bold} indicates statistical significance ($p < 0.05$) via paired bootstrap.}")
    print(r"\label{tab:pairwise_matrix}")
    print(r"\end{table*}")
    print("="*60)