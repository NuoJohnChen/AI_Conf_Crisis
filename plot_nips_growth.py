#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Set font to avoid character issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_nips_growth():
    """
    Plot NIPS/NeurIPS growth trends from 2015-2024 in a single chart
    """
    # Years
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    
    # Submission data
    submissions = np.array([1838, 2403, 3240, 4856, 6743, 9454, 9122, 10411, 12343, 15671])
    
    # Acceptance data  
    acceptances = np.array([403, 569, 678, 1011, 1428, 1900, 2344, 2672, 3218, 4037])
    
    # Rejection data (submissions - acceptances)
    rejections = submissions - acceptances
    
    # Total attendees data
    total_attendees = np.array([3852, 5231, 8008, 8648, 13000, 22823, 17091, 15390, 16382, 19756])
    
    # Online attendees (for hybrid/virtual years)
    online_attendees = np.array([0, 0, 0, 0, 0, 22823, 17091, 5555, 3075, 2978])
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(7*1.02,5*1.02))
    
    # Plot all data series in one chart
    ax.plot(years, submissions, marker='o', linewidth=3, markersize=8, 
            color='#1f77b4', label='Submissions')
    ax.plot(years, acceptances, marker='s', linewidth=3, markersize=8, 
            color='#2ca02c', label='Acceptances')
    ax.plot(years, rejections, marker='^', linewidth=3, markersize=8, 
            color='#808080', label='Rejections')  # Gray color
    ax.plot(years, total_attendees, marker='D', linewidth=3, markersize=8, 
            color='#ff7f0e', label='Total Attendees')
    
    # Online attendees with lighter color and only for years with data
    virtual_years = years[online_attendees > 0]
    virtual_online = online_attendees[online_attendees > 0]
    ax.plot(virtual_years, virtual_online, marker='v', linewidth=2, markersize=6, 
            color='#ffb366', linestyle='--', alpha=0.8, label='Online Attendees')  # Lighter orange
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('NIPS/NeurIPS Growth Trends: Submissions, Acceptances, Rejections, and Attendance (2015-2024)', 
                 fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2014.5, 2024.5)
    
    # Add annotations for conference format changes
    format_annotations = {
        2020: 'Virtual',
        2021: 'Virtual', 
        2022: 'Hybrid',
        2023: 'Hybrid',
        2024: 'Hybrid'
    }
    
    # for year, format_type in format_annotations.items():
    #     idx = np.where(years == year)[0][0]
    #     # Position annotation above the highest value for that year
    #     max_value = max(submissions[idx], total_attendees[idx])
    #     ax.annotate(format_type, 
    #                xy=(year, max_value), 
    #                xytext=(0, 20),
    #                textcoords='offset points',
    #                ha='center',
    #                fontsize=9,
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add text box with key statistics
    # stats_text = f"""Key Statistics (2015-2024):
    # • Submissions: {submissions[0]:,} → {submissions[-1]:,} ({((submissions[-1]/submissions[0])-1)*100:.0f}% growth)
    # • Acceptances: {acceptances[0]:,} → {acceptances[-1]:,} ({((acceptances[-1]/acceptances[0])-1)*100:.0f}% growth)
    # • 2024 Acceptance Rate: {(acceptances[-1]/submissions[-1])*100:.1f}%
    # • Peak Attendance: {max(total_attendees):,} (2020, virtual)"""
    stats_text = f"""Key Statistics (2015-2024):
• Submissions: {submissions[0]:,}→{submissions[-1]:,}
• Acceptances: {acceptances[0]:,}→{acceptances[-1]:,}
• 2020-2021: Virtual
• 2022-2024: Hybrid"""
    

    ax.text(0.02, 0.65, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'nips_combined_growth_trends_2015_2024.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'NIPS/NeurIPS combined growth trends chart saved as: {output_file}')
    
    # Print detailed statistics
    print("\nNIPS/NeurIPS Growth Statistics (2015-2024):")
    print(f"Submissions growth: {submissions[0]:,} -> {submissions[-1]:,} ({((submissions[-1]/submissions[0])-1)*100:.1f}% increase)")
    print(f"Acceptances growth: {acceptances[0]:,} -> {acceptances[-1]:,} ({((acceptances[-1]/acceptances[0])-1)*100:.1f}% increase)")
    print(f"Rejections growth: {rejections[0]:,} -> {rejections[-1]:,} ({((rejections[-1]/rejections[0])-1)*100:.1f}% increase)")
    print(f"Peak attendance: {max(total_attendees):,} (2020, virtual)")
    print(f"2024 acceptance rate: {(acceptances[-1]/submissions[-1])*100:.1f}%")
    
    plt.show()

def plot_detailed_breakdown():
    """
    Plot a more detailed breakdown with additional metrics
    """
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    submissions = np.array([1838, 2403, 3240, 4856, 6743, 9454, 9122, 10411, 12343, 15671])
    acceptances = np.array([403, 569, 678, 1011, 1428, 1900, 2344, 2672, 3218, 4037])
    
    # Calculate acceptance rates
    acceptance_rates = (acceptances / submissions) * 100
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5,6))
    
    # 1. Absolute numbers
    ax1.plot(years, submissions, 'o-', linewidth=2, label='Submissions', color='#1f77b4')
    ax1.plot(years, acceptances, 's-', linewidth=2, label='Acceptances', color='#2ca02c') 
    ax1.set_title('Submissions vs Acceptances')
    ax1.set_ylabel('Number of Papers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Acceptance rate
    ax2.plot(years, acceptance_rates, 'D-', linewidth=2, color='#ff7f0e', markersize=6)
    ax2.set_title('Acceptance Rate Trend')
    ax2.set_ylabel('Acceptance Rate (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(15, 30)
    
    # 3. Year-over-year growth
    submission_growth = np.diff(submissions) / submissions[:-1] * 100
    acceptance_growth = np.diff(acceptances) / acceptances[:-1] * 100
    
    ax3.bar(years[1:]-0.2, submission_growth, 0.4, label='Submissions Growth', alpha=0.7)
    ax3.bar(years[1:]+0.2, acceptance_growth, 0.4, label='Acceptances Growth', alpha=0.7)
    ax3.set_title('Year-over-Year Growth Rate')
    ax3.set_ylabel('Growth Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Cumulative papers
    cumulative_submissions = np.cumsum(submissions)
    cumulative_acceptances = np.cumsum(acceptances)
    
    ax4.plot(years, cumulative_submissions, 'o-', linewidth=2, label='Cumulative Submissions')
    ax4.plot(years, cumulative_acceptances, 's-', linewidth=2, label='Cumulative Acceptances')
    ax4.set_title('Cumulative Papers Over Time')
    ax4.set_ylabel('Cumulative Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Set x-axis labels for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Year')
        ax.set_xlim(2014.5, 2024.5)
    
    plt.tight_layout()
    
    # Save the detailed plot
    output_file = 'nips_detailed_analysis_2015_2024.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Detailed NeurIPS analysis saved as: {output_file}')
    
    plt.show()

def plot_offline_attendance_prediction():
    """
    Plot offline attendance trends with realistic prediction to 2035
    Using all historical data (excluding 2020-2021 virtual years) for extrapolation
    """
    # Historical data (excluding 2020-2021 virtual years)
    years_data = np.array([2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024])
    
    # Offline attendance data
    offline_attendance = np.array([
        3852,   # 2015
        5231,   # 2016  
        8008,   # 2017
        8648,   # 2018
        13000,  # 2019
        9835,   # 2022: 15390 - 5555 (post-pandemic recovery)
        13307,  # 2023: 16382 - 3075
        16777   # 2024: 19756 - 2978
    ])
    
    # Create prediction years 2025-2035
    prediction_years = np.arange(2025, 2036)
    
    # Conservative growth model (slower growth rate)
    # Assume growth rate decreases over time (saturation effect)
    conservative_predictions = []
    current_attendance = offline_attendance[-1]  # 2024 value
    annual_growth_rate = 0.20  # Start with 20% growth, decreasing
    
    for i, year in enumerate(prediction_years):
        # Decreasing growth rate over time
        growth_rate = annual_growth_rate * (0.95 ** i)  # 5% decrease each year
        current_attendance = current_attendance * (1 + growth_rate)
        conservative_predictions.append(current_attendance)
    
    conservative_predictions = np.array(conservative_predictions)
    
    # Combine historical data and predictions for continuous curve
    all_years = np.concatenate([years_data, prediction_years])
    all_attendance = np.concatenate([offline_attendance, conservative_predictions])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7.5,5))
    
    # Plot historical data points
    ax.plot(years_data, offline_attendance, marker='o', linewidth=3, markersize=10, 
            color='#1f77b4', label='Historical Offline Attendance')
    
    # Plot complete curve from 2015 to 2035 (historical + predictions)
    ax.plot(all_years, all_attendance, linewidth=2, 
            color='#2ca02c', linestyle=':', alpha=0.8, label='Conservative Growth Prediction')
    
    # Plot prediction points with markers
    ax.plot(prediction_years, conservative_predictions, marker='^', linewidth=0, markersize=6, 
            color='#2ca02c', alpha=0.8)
    
    # Add horizontal line at 18,000 covering entire range
    ax.axhline(y=18000, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label='18,000 Target')
    
    # Add horizontal line at 80,000 covering entire range
    ax.axhline(y=80000, color='purple', linestyle='--', linewidth=2, alpha=0.7, 
               label='80,000 Target')
    
    # # Annotations for excluded years - moved down to avoid 80,000 line
    # ax.annotate('2020-2021 Virtual\n(Data Excluded)', 
    #             xy=(2020.5, 65000), 
    #             fontsize=10,
    #             ha='center',
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Mark when 18,000 might be reached for conservative model
    if any(conservative_predictions >= 18000):
        crossing_year = prediction_years[conservative_predictions >= 18000][0]
        crossing_value = conservative_predictions[conservative_predictions >= 18000][0]
        # ax.annotate(f'18K reached\nConservative: {crossing_year}', 
        #            xy=(crossing_year, crossing_value), 
        #            xytext=(10, 15),
        #            textcoords='offset points',
        #            fontsize=9,
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
        #            arrowprops=dict(arrowstyle='->', color='#2ca02c'))
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Offline Attendance Count', fontsize=12)
    ax.set_title('NIPS/NeurIPS Offline Attendance: Historical Data and Growth Predictions (2015-2035)', 
                 fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, 0.85))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2014, 2036)
    ax.set_ylim(0, max(conservative_predictions) * 1.1)
    
    # Set x-axis ticks to show every year
    ax.set_xticks(np.arange(2015, 2036, 1))
    ax.tick_params(axis='x', rotation=45)
    
    # Add statistics text box - moved to left center
    growth_2015_2024 = ((offline_attendance[-1] / offline_attendance[0]) - 1) * 100
    stats_text = f"""Historical Trend (2015-2019, 2022-2024):
    • 2015: {offline_attendance[0]:,} attendees
    • 2024: {offline_attendance[-1]:,} attendees  
    • 2020-2021: Virtual
    • Conservative 2035: {conservative_predictions[-1]:,.0f}"""
    
    ax.text(0.02, 0.4, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'nips_offline_attendance_realistic_prediction.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'NIPS/NeurIPS realistic offline attendance prediction saved as: {output_file}')
    
    # Print detailed predictions
    print(f"\nConservative Growth Predictions:")
    for i, year in enumerate(prediction_years):
        print(f"{year}: {conservative_predictions[i]:,.0f} attendees")
    
    # Check 18K milestone
    conservative_18k = prediction_years[conservative_predictions >= 18000]
    
    if len(conservative_18k) > 0:
        print(f"\n18,000 milestone - Conservative model: {conservative_18k[0]}")
    
    plt.show()

if __name__ == "__main__":
    print("Generating NIPS/NeurIPS realistic offline attendance prediction...")
    # plot_offline_attendance_prediction() 
    plot_nips_growth()