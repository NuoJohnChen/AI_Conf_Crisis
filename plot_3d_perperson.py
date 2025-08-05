#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats

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

def aggregate_by_parent(df, metric='contribution_per_person'):
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

def plot_3d_perperson(csv_file="iclr_by_year_contribution_per_person.csv"):
    """
    Plot a 3D line chart showing how the contribution_per_person metric changes with year and research area (grouped by parent)
    """
    # Read data
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    required_columns = ['Year', 'Area', 'contribution_per_person']
    if not all(col in df.columns for col in required_columns):
        print("Error: CSV file missing required columns (Year, Area, contribution_per_person)")
        return
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df)
    
    if df_aggregated.empty:
        print("Error: No data after aggregation, please check data format")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Order parent categories
    parent_order = ['Theory', 'Systems', 'Interdisciplinary', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())
    
    # Draw a 3D line for each parent category
    colors = plt.cm.tab10(np.linspace(0, 1, len(parents)))
    
    # Set spacing
    spacing = 0.5
    
    for i, parent in enumerate(parents):
        parent_data = df_aggregated[df_aggregated['Parent'] == parent].sort_values('Year')
        
        if len(parent_data) > 0:
            x = parent_data['Year'].values
            y = [i * spacing] * len(x)
            z = parent_data['contribution_per_person'].values
            
            # Plot 3D line
            ax.plot(x, y, z, 
                   marker='o', 
                   linewidth=3, 
                   markersize=8, 
                   color=colors[i], 
                   label=parent)
            
            # Calculate z-axis range for projection
            if all_z_values:
                z_min_proj = min(all_z_values)
            else:
                z_min_proj = 0
            
            # Draw projection line to bottom plane
            ax.plot(x, y, [z_min_proj] * len(x), 
                   '--', 
                   linewidth=1, 
                   color=colors[i], 
                   alpha=0.3)
            
            # Draw vertical lines from data points to bottom plane
            for j in range(len(x)):
                ax.plot([x[j], x[j]], [y[j], y[j]], [z_min_proj, z[j]], 
                       ':', 
                       linewidth=1, 
                       color=colors[i], 
                       alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Research Area Category', fontsize=12)
    ax.set_zlabel('contribution_per_person', fontsize=12)
    
    # Set y-axis tick labels and range
    ax.set_yticks([i * spacing for i in range(len(parents))])
    ax.set_yticklabels(parents)
    ax.set_ylim(-0.2, (len(parents)-1) * spacing + 0.2)
    
    # Set z-axis range to actual data range
    all_z_values = []
    for parent in parents:
        parent_data = df_aggregated[df_aggregated['Parent'] == parent]
        if not parent_data.empty:
            all_z_values.extend(parent_data['contribution_per_person'].values)
    if all_z_values:
        z_min, z_max = min(all_z_values), max(all_z_values)
        z_range = z_max - z_min
        ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
    
    # Set title
    ax.set_title('Change of contribution_per_person by Year and Research Area Category', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    output_file = 'contribution_per_person_3d_plot_by_parent.png'
    plt.savefig(output_file, dpi=300)
    print(f"3D chart saved as: {os.path.abspath(output_file)}")
    
    # Show chart
    plt.show()

def plot_3d_surface(csv_file="iclr_by_year_contribution_per_person.csv"):
    """
    Plot a 3D surface chart showing how the contribution_per_person metric changes with year and research area (grouped by parent)
    """
    # Read data
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    required_columns = ['Year', 'Area', 'contribution_per_person']
    if not all(col in df.columns for col in required_columns):
        print("Error: CSV file missing required columns (Year, Area, contribution_per_person)")
        return
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df)
    
    if df_aggregated.empty:
        print("Error: No data after aggregation, please check data format")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Order parent categories
    parent_order = ['Theory', 'Systems', 'Interdisciplinary', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())
    
    # Set spacing
    spacing = 0.5
    
    # Create grid data
    X, Y = np.meshgrid(years, [i * spacing for i in range(len(parents))])
    Z = np.zeros((len(parents), len(years)))
    
    # Fill Z data
    for i, parent in enumerate(parents):
        for j, year in enumerate(years):
            data_point = df_aggregated[(df_aggregated['Parent'] == parent) & (df_aggregated['Year'] == year)]
            if not data_point.empty:
                Z[i, j] = data_point['contribution_per_person'].iloc[0]
            else:
                Z[i, j] = np.nan
    
    # Plot 3D surface
    surf = ax.plot_surface(X, Y, Z, 
                          cmap='viridis', 
                          alpha=0.8,
                          linewidth=0,
                          antialiased=True)
    
    # Set axis labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Research Area Category', fontsize=12)
    ax.set_zlabel('contribution_per_person', fontsize=12)
    
    # Set y-axis tick labels and range
    ax.set_yticks([i * spacing for i in range(len(parents))])
    ax.set_yticklabels(parents)
    ax.set_ylim(-0.2, (len(parents)-1) * spacing + 0.2)
    
    # Set z-axis range to actual data range
    valid_z_values = Z[~np.isnan(Z)]
    if len(valid_z_values) > 0:
        z_min, z_max = np.min(valid_z_values), np.max(valid_z_values)
        z_range = z_max - z_min
        ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
    
    # Set title
    ax.set_title('3D Surface Plot of contribution_per_person by Research Category', fontsize=14, pad=20)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    output_file = 'contribution_per_person_3d_surface_by_parent.png'
    plt.savefig(output_file, dpi=300)
    print(f"3D surface chart saved as: {os.path.abspath(output_file)}")
    
    # Show chart
    plt.show()

def plot_3d_perperson_separate(csv_file="iclr_by_year_contribution_per_person.csv"):
    """
    Plot a separate 3D subplot for each parent category, showing how the contribution_per_person changes with year
    """
    df = pd.read_csv(csv_file)
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df)
    
    # Order parent categories
    parent_order = ['Theory', 'Systems', 'Interdisciplinary', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())

    # Color mapping
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(parents))]

    fig = plt.figure(figsize=(5 * len(parents), 6))
    for i, parent in enumerate(parents):
        ax = fig.add_subplot(1, len(parents), i+1, projection='3d')
        parent_data = df_aggregated[df_aggregated['Parent'] == parent].sort_values('Year')
        x = parent_data['Year'].values
        y = np.zeros_like(x)
        z = parent_data['contribution_per_person'].values

        # Curve
        ax.plot(x, y, z, marker='o', linewidth=2, markersize=6, color=colors[i], label=parent)
        
        # Calculate projection bottom position
        if len(z) > 0:
            z_min_proj = min(z)
        else:
            z_min_proj = 0
        
        # Projection
        ax.plot(x, y, np.full_like(z, z_min_proj), '--', color=colors[i], alpha=0.5)
        for j in range(len(x)):
            ax.plot([x[j], x[j]], [y[j], y[j]], [z_min_proj, z[j]], ':', color=colors[i], alpha=0.5)

        # Shadow under the curve
        if len(x) > 0:
            verts = [list(zip(x, y, z)), list(zip(x[::-1], y[::-1], np.full_like(z, z_min_proj)))]
            poly = Poly3DCollection([verts[0] + verts[1]], facecolors=colors[i], alpha=0.18)
            ax.add_collection3d(poly)

        ax.set_xlabel('Year')
        ax.set_ylabel('')
        ax.set_zlabel('contribution_per_person')
        ax.set_title(parent)
        ax.set_yticks([])
        
        # Set z-axis range to the data range of this parent
        if len(z) > 0:
            z_min, z_max = min(z), max(z)
            z_range = z_max - z_min
            if z_range > 0:
                ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
            else:
                ax.set_zlim(z_min - 0.1, z_max + 0.1)

    plt.tight_layout()
    output_file = 'contribution_per_person_3d_separate_by_parent.png'
    plt.savefig(output_file, dpi=300)
    print(f'Separate 3D charts saved as: {os.path.abspath(output_file)}')
    plt.show()

def plot_3d_publicationcount_together(csv_file="iclr_by_year_contribution_per_person.csv", metric="PublicationCount"):
    """
    Plot all parent categories on one 3D chart, showing how the specified metric changes with year, including trend prediction to 2035
    """
    df = pd.read_csv(csv_file)
    
    # Aggregate data by parent
    df_aggregated = aggregate_by_parent(df, metric)
    
    # Order parent categories
    parent_order = ['Theory', 'Systems', 'Interdisciplinary', 'AI']
    parents = [p for p in parent_order if p in df_aggregated['Parent'].unique()]
    years = sorted(df_aggregated['Year'].unique())

    # Color mapping
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(parents))]

    fig = plt.figure(figsize=(14,10))  # Increase figure size, especially width for z-axis label space
    ax = fig.add_subplot(111, projection='3d')

    spacing = 0.5
    
    # Collect all z values for subsequent calculations
    all_z_values = []
    ai_prediction_2030 = 0  # To store the prediction value for AI in 2030
    
    for parent in parents:
        parent_data = df_aggregated[df_aggregated['Parent'] == parent]
        if not parent_data.empty:
            all_z_values.extend(parent_data[metric].values)
    
    # Calculate projection bottom position (set to 0, ensuring xy-axis corresponds to publication count of 0)
    z_min_proj = 0
    
    # Extend years to 2030 for prediction
    future_years = np.arange(max(years) + 1, 2031)
    all_years_extended = list(years) + list(future_years)
    
    for i, parent in enumerate(parents):
        parent_data = df_aggregated[df_aggregated['Parent'] == parent].sort_values('Year')
        x = parent_data['Year'].values
        y = np.ones_like(x) * i * spacing
        z = parent_data[metric].values

        if len(x) > 0:
            # Historical data curve
            ax.plot(x, y, z, marker='o', linewidth=3, markersize=8, color=colors[i], label=parent)
            
            # Trend fitting and prediction (only for non-contribution_per_person metrics)
            if metric != 'contribution_per_person' and len(x) >= 3:  # Need at least 3 data points for fitting
                # Use log-linear regression (exponential fit)
                log_z = np.log(z)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_z)
                
                # Exponential function
                def exp_func(x_new):
                    return np.exp(slope * x_new + intercept)
                
                # Predict values for future years
                future_z = exp_func(future_years)
                # Ensure predicted values are not negative
                future_z = np.maximum(future_z, 0)
                
                # If it's the AI category, save the prediction value for 2030 and R² value
                if parent == 'AI':
                    ai_prediction_2030 = exp_func(2030)
                    ai_prediction_2030 = max(ai_prediction_2030, 0)  # Ensure it's not negative
                    r_squared = r_value**2
                    growth_rate = (np.exp(slope) - 1) * 100
                    print(f"AI {metric} log-linear regression results:")
                    print(f"  Growth rate: {growth_rate:.2f}% annual growth")
                    print(f"  R²: {r_squared:.4f}")
                    print(f"  2030 prediction: {ai_prediction_2030:.0f}")
                
                future_y = np.ones_like(future_years) * i * spacing
                
                # Create continuous prediction curve (including connection from the last historical data point to the first prediction point)
                # Connect the last historical data point and the first prediction point
                if len(x) > 0 and len(future_years) > 0:
                    connect_x = [x[-1], future_years[0]]
                    connect_y = [y[-1], future_y[0]]
                    connect_z = [z[-1], future_z[0]]
                    ax.plot(connect_x, connect_y, connect_z, 
                           linestyle='--', linewidth=2, color=colors[i], alpha=0.7)
                
                # Plot prediction curve (dashed line)
                ax.plot(future_years, future_y, future_z, 
                       linestyle='--', linewidth=2, color=colors[i], alpha=0.7,
                       label=f'{parent} (Prediction)')
                
                # Shadow for prediction part (including connection)
                if len(future_years) > 0 and len(x) > 0:
                    # Shadow from the last historical data point to the prediction data
                    extended_x = [x[-1]] + list(future_years)
                    extended_y = [y[-1]] + list(future_y)
                    extended_z = [z[-1]] + list(future_z)
                    verts_pred = [list(zip(extended_x, extended_y, extended_z)), 
                                 list(zip(extended_x[::-1], extended_y[::-1], np.full_like(extended_z, z_min_proj)))]
                    poly_pred = Poly3DCollection([verts_pred[0] + verts_pred[1]], facecolors=colors[i], alpha=0.12)
                    ax.add_collection3d(poly_pred)
            
            # Projection (only historical data)
            ax.plot(x, y, np.full_like(z, z_min_proj), '--', color=colors[i], alpha=0.3)
            for j in range(len(x)):
                ax.plot([x[j], x[j]], [y[j], y[j]], [z_min_proj, z[j]], ':', color=colors[i], alpha=0.3)

            # Shadow under the curve (only historical data)
            verts = [list(zip(x, y, z)), list(zip(x[::-1], y[::-1], np.full_like(z, z_min_proj)))]
            poly = Poly3DCollection([verts[0] + verts[1]], facecolors=colors[i], alpha=0.18)
            ax.add_collection3d(poly)

    # Set axis labels, increase font size and add more spacing
    ax.set_xlabel('Year', fontsize=16, labelpad=25)
    ax.set_ylabel('Research Area Category', fontsize=16, labelpad=40)  # Further increase labelpad
    ax.set_zlabel(metric, fontsize=16, labelpad=35)  # Increase z-axis label spacing
    
    # Set y-axis ticks and labels, increase font size, Interdisciplinary wrap display
    ax.set_yticks([i * spacing for i in range(len(parents))])
    display_parents = [p if p != 'Interdisciplinary' else 'Interdis-\nciplinary' for p in parents]
    ax.set_yticklabels(display_parents, fontsize=14)
    ax.set_ylim(-0.2, (len(parents)-1) * spacing + 0.2)
    
    # Set x-axis range, based on metric decide whether to include prediction years
    if metric == 'contribution_per_person':
        ax.set_xlim(min(years), max(years))  # Only show historical data range
        x_ticks = range(min(years), max(years) + 1, 2)  # Two years interval
    else:
        ax.set_xlim(min(years), 2030)  # Include prediction years
        x_ticks = range(min(years), 2031, 2)  # Two years interval
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=14, rotation=15)  # Set 15 degree tilt
    ax.tick_params(axis='z', labelsize=14, pad=15)  # Increase z-axis tick number spacing
    ax.tick_params(axis='y', labelsize=14, pad=10)  # Increase y-axis tick number spacing
    
    # Set z-axis range, max value using AI's 2030 prediction, min value set to 0
    if all_z_values:
        z_min = 0  # Bottom plane set to 0
        # Use AI's 2030 prediction as max value, if no prediction value, use historical data max value
        z_max = ai_prediction_2030 if ai_prediction_2030 > 0 else max(all_z_values)
        z_range = z_max - z_min
        ax.set_zlim(z_min, z_max + 0.1 * z_range)
        
        print(f"Z-axis range: min = {z_min:.0f}, max = {z_max:.0f} (AI 2030 prediction)")
    else:
        ax.set_zlim(0, ai_prediction_2030 * 1.1 if ai_prediction_2030 > 0 else 1000)
    
    # Remove vertical separator
    # max_year = max(years)
    # for i in range(len(parents)):
    #     ax.plot([max_year, max_year], [i * spacing, i * spacing], 
    #            [ax.get_zlim()[0], ax.get_zlim()[1]], 
    #            color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # ax.set_title(f'{metric} by Research Category (with 2035 Trend Prediction)', fontsize=18, pad=20)
    # Remove legend
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # Adjust figure margins, increase right and bottom margins
    # plt.subplots_adjust(left=0.2, right=0.65, top=0.9, bottom=0.2)
    # plt.subplots_adjust(left=0.15, right=0.83, top=0.95, bottom=0.25)
    
    # Generate different file names based on metric
    if metric == 'contribution_per_person':
        output_file = 'contribution_per_person_3d_together_by_parent111.png'
    else:
        output_file = f'{metric.lower()}_3d_together_by_parent_with_prediction111.png'
    plt.savefig(output_file, dpi=300)
    print(f'3D plot with prediction saved as: {os.path.abspath(output_file)}')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot 3D charts for PerPerson/PublicationCount metrics")
    parser.add_argument(
        "--input",
        type=str,
        default="iclr_by_year_contribution_per_person.csv",
        help="Input CSV file path (default: iclr_by_year_contribution_per_person.csv)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['line', 'surface', 'both', 'separate', 'together'],
        default='both',
        help="Chart type: line (3D line chart), surface (3D surface chart), both (draw both), separate (each area in a subplot), together (all areas in one chart)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=['PublicationCount', 'PerPerson', 'contribution_per_person', 'ICLRPoint'],
        default='contribution_per_person',
        help="Metric to visualize (default: contribution_per_person)"
    )
    
    args = parser.parse_args()
    
    if args.plot_type in ['line', 'both']:
        print("Plotting 3D line chart...")
        plot_3d_perperson(args.input)
    
    if args.plot_type in ['surface', 'both']:
        print("Plotting 3D surface chart...")
        plot_3d_surface(args.input)
    
    if args.plot_type == 'separate':
        print("Plotting separate 3D charts for each category...")
        plot_3d_perperson_separate(args.input)
    
    if args.plot_type == 'together':
        print(f"Plotting 3D chart for all categories (metric: {args.metric})...")
        plot_3d_publicationcount_together(args.input, args.metric) 