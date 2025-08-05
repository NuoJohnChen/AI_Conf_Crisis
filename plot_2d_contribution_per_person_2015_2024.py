#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font to avoid character issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

area_to_parent = {
    # AI
    "ai": "AI",
    "vision": "AI",
    "mlmining": "AI",
    "nlp": "AI",
    "inforet": "AI",
    
    # Systems
    "arch": "Systems",
    "comm": "Systems",
    "sec": "Systems",
    "mod": "Systems",
    "da": "Systems",
    "bed": "Systems",
    "hpc": "Systems",
    "mobile": "Systems",
    "metrics": "Systems",
    "ops": "Systems",
    "plan": "Systems",
    "soft": "Systems",
    
    # Theory
    "act": "Theory",
    "crypt": "Theory",
    "log": "Theory",
    
    # Interdisciplinary Areas
    "bio": "Interdisciplinary",
    "graph": "Interdisciplinary",
    "csed": "Interdisciplinary",
    "ecom": "Interdisciplinary",
    "chi": "Interdisciplinary",
    "robotics": "Interdisciplinary",
    "visualization": "Interdisciplinary"
}

parent_to_area = {
    "AI": ["ai", "vision", "mlmining", "nlp", "inforet"],
    "Systems": ["arch", "comm", "sec", "mod", "da", "bed", "hpc", "mobile", "metrics", "ops", "plan", "soft"],
    "Theory": ["act", "crypt", "log"],
    "Interdisciplinary": ["bio", "graph", "csed", "ecom", "chi", "robotics", "visualization"]
}

def aggregate_by_parent(df, metric='contribution_per_person_not_adjusted'):
    """
    Aggregate data by parent category. For count metrics, sum; for per person metrics, average.
    """
    # Add parent column
    df['Parent'] = df['Area'].map(area_to_parent)
    
    # Filter out data without parent mapping
    df_with_parent = df[df['Parent'].notna()].copy()
    
    # Choose aggregation method based on metric type
    if metric in ['PublicationCount', 'EffectiveFaculties', 'ICLRPoint']:
        # For count metrics, use sum
        df_aggregated = df_with_parent.groupby(['Year', 'Parent']).agg({
            metric: 'sum'
        }).reset_index()
    else:
        # For per person metrics, use mean
        df_aggregated = df_with_parent.groupby(['Year', 'Parent']).agg({
            metric: 'mean'
        }).reset_index()
    
    return df_aggregated

def plot_2d_contribution_per_person_2015_2024(csv_file="iclr_by_year_not_adjusted_2015_2024.csv"):
    """
    Plot 2D chart with year on x-axis and contribution_per_person on y-axis
    Display lines for four research areas with legend in upper left corner
    """
    df = pd.read_csv(csv_file)
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df, 'contribution_per_person_not_adjusted')
    
    # Order parent categories: Interdisciplinary, Systems, Theory, AI
    parent_order = ['Interdisciplinary', 'Systems', 'Theory', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())

    # Color mapping
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(parents))]

    # Create figure
    fig, ax = plt.subplots(figsize=(7.5,5))

    # Plot lines for each research area
    for i, parent in enumerate(parents):
        parent_data = df_aggregated[df_aggregated['Parent'] == parent].sort_values('Year')
        x = parent_data['Year'].values
        y = parent_data['contribution_per_person_not_adjusted'].values

        if len(x) > 0:
            # Plot line
            ax.plot(x, y, marker='o', linewidth=3, markersize=8, 
                   color=colors[i], label=parent)

    # Set axis labels
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('contribution_per_person', fontsize=14)
    
    # Set x-axis range - 2015 to 2024
    ax.set_xlim(2015, 2024)
    x_ticks = range(2015, 2025, 1)  # One interval per year
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set grid
    ax.grid(True, alpha=0.3)
    
    # Set legend in upper left corner
    ax.legend(loc='upper left', fontsize=12, frameon=True, 
              fancybox=True, shadow=True)
    
    # Set title
    ax.set_title('contribution_per_person (2015-2024)\nUsing count instead of adjustedcount', 
                 fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    output_file = 'contribution_per_person_2d_2015_2024.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'2D plot saved as: {os.path.abspath(output_file)}')
    plt.show()

def plot_2d_contribution_per_person_with_confidence(csv_file="iclr_by_year_not_adjusted_2015_2024.csv"):
    """
    Plot 2D chart with confidence intervals (standard deviation)
    """
    df = pd.read_csv(csv_file)
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df, 'contribution_per_person_not_adjusted')
    
    # Order parent categories: Interdisciplinary, Systems, Theory, AI
    parent_order = ['Interdisciplinary', 'Systems', 'Theory', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())

    # Color mapping
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(parents))]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot lines for each research area
    for i, parent in enumerate(parents):
        parent_data = df_aggregated[df_aggregated['Parent'] == parent].sort_values('Year')
        x = parent_data['Year'].values
        y = parent_data['contribution_per_person_not_adjusted'].values

        if len(x) > 0:
            # Plot line
            ax.plot(x, y, marker='o', linewidth=3, markersize=8, 
                   color=colors[i], label=parent)
            
            # Add data point labels (optional)
            for j, (year, value) in enumerate(zip(x, y)):
                ax.annotate(f'{value:.2f}', (year, value), 
                           textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)

    # Set axis labels
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('contribution_per_person', fontsize=14)
    
    # Set x-axis range - 2015 to 2024
    ax.set_xlim(2015, 2024)
    x_ticks = range(2015, 2025, 1)  # One interval per year
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set grid
    ax.grid(True, alpha=0.3)
    
    # Set legend in upper left corner
    ax.legend(loc='upper left', fontsize=12, frameon=True, 
              fancybox=True, shadow=True)
    
    # Set title
    ax.set_title('contribution_per_person (2015-2024)\nUsing count instead of adjustedcount', 
                 fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    output_file = 'contribution_per_person_2d_with_labels_2015_2024.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'2D plot with labels saved as: {os.path.abspath(output_file)}')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot 2D charts for contribution_per_person metrics (2015-2024)")
    parser.add_argument(
        "--input",
        type=str,
        default="iclr_by_year_not_adjusted_2015_2024.csv",
        help="Input CSV file path (default: iclr_by_year_not_adjusted_2015_2024.csv)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['basic', 'with_labels', 'both'],
        default='basic',
        help="Chart type: basic (basic chart), with_labels (with value labels), both (draw both)"
    )
    
    args = parser.parse_args()
    
    if args.plot_type in ['basic', 'both']:
        print("Plotting 2D chart for contribution_per_person (2015-2024)...")
        plot_2d_contribution_per_person_2015_2024(args.input)
    
    if args.plot_type in ['with_labels', 'both']:
        print("Plotting 2D chart with labels (2015-2024)...")
        plot_2d_contribution_per_person_with_confidence(args.input) 