import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Data based on the research analysis, in tonnes of CO2 equivalent (tCO2e).
# The years and host cities correspond to the data points.
years = ['2023\nKigali', '2024\nVienna', '2025\nSingapore']
emissions_data = {
    'Venue Operations': np.array([19.5, 26.9, 43.7]),
    'Meals': np.array([38.0, 52.5, 85.0]),
    'Accommodation': np.array([188.3, 129.4, 370.0]),
    'Air Travel': np.array([2330.1, 2219.2, 4199.6])
}
categories = list(emissions_data.keys())
data = np.array(list(emissions_data.values()))

# --- Calculations ---
# Calculate total emissions for each year to find percentages for labels.
totals = np.sum(data, axis=0)

# --- Colors and Styling ---
# Using a gradient color palette from light to dark from bottom to top
colors = ['#d5e6fb', '#9dd1fb','#459af8','#b190f8'] # Light blue, Light cyan, Blue, Purple
plt.style.use('seaborn-v0_8-whitegrid')

# --- Plotting ---
fig, ax = plt.subplots(figsize=(9,6)) # Figure size for grouped bars

# Set up the positions for grouped bars
x = np.arange(len(years))
width = 0.2  # Width of each bar

# Create background bars for totals
background_bars = ax.bar(x + width * 1.5, totals, width * 4, color='lightgray', alpha=0.3)

# Create the stacked bars within each background bar
for i, category in enumerate(categories):
    bar_positions = x + i * width
    bars = ax.bar(bar_positions, data[i], width, label=category, color=colors[i])
    
    # Add individual values inside each bar near the top
    for j, rect in enumerate(bars):
        height = rect.get_height()
        if height > 0:
            # Special formatting for Air Travel (category index 3)
            if i == 3:  # Air Travel
                value_str = f'{height:,.1f}'
                if '.' in value_str:
                    parts = value_str.split('.')
                    formatted_value = f'{parts[0]}\n.{parts[1]}'
                else:
                    formatted_value = value_str
                # Position Air Travel text slightly lower
                text_y = rect.get_y() + height * 0.7
            else:
                formatted_value = f'{height:,.1f}'
                text_y = rect.get_y() + height * 0.8
            
            ax.text(rect.get_x() + rect.get_width() / 2., 
                    text_y,  # Position inside the bar near the top
                    formatted_value, 
                    ha='center', 
                    va='center', 
                    fontsize=8, 
                    fontweight='bold',
                    color='white')

# Add total values on top of background bars
for j, total in enumerate(totals):
    ax.text(x[j] + width * 1.5,  # Center of the background bar
            total + total * 0.03,  # Position higher above the background bar
            f'Total\n{total:,.1f}', 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold')

# --- Set x-axis labels and positions ---
ax.set_xticks(x + width * 1.5)  # Center the labels under the background bars
ax.set_xticklabels(years)

# --- Titles and Labels ---
ax.set_title('ICLR Conference Carbon Emissions by Accepted Papers (2023-2025)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Carbon Emissions (tCO2e)', fontsize=12, labelpad=15)
ax.set_xlabel('Conference Year and Host City', fontsize=12, labelpad=15)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, ncol=4, columnspacing=0.5, bbox_to_anchor=(-0.01, 1.02))

# --- Improve layout and formatting ---
# Set y-axis to logarithmic scale
ax.set_yscale('log')
ax.set_ylim(bottom=1, top=10000)  # Set range from 1 to 10^4

# Set y-axis ticks to powers of 10
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# Adjust layout to prevent the legend from being cut off.
plt.tight_layout()

# --- Save and Show ---
# Save the figure for use in publications.
plt.savefig('ICLR_Emissions_Graph_Corrected.png', dpi=300, bbox_inches='tight')
plt.show()
print("save image in ICLR_Emissions_Graph_Corrected.png")
