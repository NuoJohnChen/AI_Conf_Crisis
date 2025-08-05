#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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

def aggregate_by_parent(df, metric='EffectiveFaculties'):
    df['Parent'] = df['Area'].map(area_to_parent)
    df_with_parent = df[df['Parent'].notna()].copy()
    df_aggregated = df_with_parent.groupby(['Year', 'Parent']).agg({metric: 'sum'}).reset_index()
    return df_aggregated

def plot_2d_effective_faculties_by_parent(csv_file="iclr_by_year_not_adjusted_2015_2024.csv"):
    df = pd.read_csv(csv_file)
    df_agg = aggregate_by_parent(df, 'EffectiveFaculties')
    parent_order = ['Interdisciplinary', 'Systems', 'Theory', 'AI']
    parents = [p for p in parent_order if p in df_agg['Parent'].unique()]
    years = sorted(df_agg['Year'].unique())
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(parents))]
    fig, ax = plt.subplots(figsize=(7.5,5))
    for i, parent in enumerate(parents):
        pdata = df_agg[df_agg['Parent'] == parent].sort_values('Year')
        x = pdata['Year'].values
        y = pdata['EffectiveFaculties'].values
        if len(x) > 0:
            ax.plot(x, y, marker='o', linewidth=3, markersize=8, color=colors[i], label=parent)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Effective Faculties', fontsize=14)
    ax.set_xlim(2015, 2024)
    ax.set_xticks(range(2015, 2025, 1))
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.set_title('Effective Faculties by Parent Area (2015-2024)', fontsize=16, pad=20)
    plt.tight_layout()
    output_file = 'effective_faculties_2d_by_parent_2015_2024.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'2D plot saved as: {os.path.abspath(output_file)}')
    plt.show()

if __name__ == "__main__":
    plot_2d_effective_faculties_by_parent() 