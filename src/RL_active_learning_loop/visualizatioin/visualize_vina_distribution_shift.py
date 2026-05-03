import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define your file paths and reaction names
file_path_1 = "/blue/lic/huangzihang/repos/elion/src/vina/LGBM_suzuki_vina_2_no_smile.csv"
file_path_2 = "/blue/lic/huangzihang/repos/elion/src/vina/LGBM_suzuki_vina_3_no_smile.csv"
label_1 = "Initial Vina"
label_2 = "Optimized Vina"
output_name = '/blue/lic/huangzihang/repos/elion/src/RL_active_learning_loop/visualizatioin/LGBM_suzuki_2_vs_3.png'

# Column name to visualize (adjust if different in your CSVs)
label_col = 'Affinity' 

try:
    # Load the datasets
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # Create the combined plot
    # We use stat="density" to ensure the distributions are comparable even if sample sizes differ
    ax = sns.histplot(df1[label_col], kde=True, color='skyblue', label=label_1, alpha=0.5, bins=30, stat="density")
    sns.histplot(df2[label_col], kde=True, color='orange', label=label_2, alpha=0.5, bins=30, stat="density", ax=ax)

    # Calculate and plot the means
    mean1 = df1[label_col].mean()
    mean2 = df2[label_col].mean()

    # Draw vertical lines to "switch" or compare the means
    plt.axvline(mean1, color='blue', linestyle='--', linewidth=2, label=f'Mean {label_1}: {mean1:.2f}')
    plt.axvline(mean2, color='darkorange', linestyle='--', linewidth=2, label=f'Mean {label_2}: {mean2:.2f}')

    # Adding titles and labels
    ax.set_title(f'Distribution Comparison: {label_1} vs {label_2}', fontsize=14)
    ax.set_xlabel(label_col, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Show legend to distinguish datasets and means
    plt.legend()

    # Save the visualization
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Visualization successfully saved as '{output_name}'")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")