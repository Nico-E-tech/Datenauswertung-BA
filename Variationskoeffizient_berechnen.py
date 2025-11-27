import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
USE_MEAN_FOR_CV = True  # Set to True to divide by mean, False to use custom value
CUSTOM_DIVISOR = 1.0    # Custom value to divide by if USE_MEAN_FOR_CV is False

# File paths
input_file = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Dunkelrauschen ohne Lichtschutz (Noise Floor)\CSV\Dunkelrauschen ohne Lichtschutz.csv")
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Dunkelrauschen ohne Lichtschutz (Noise Floor)\Variationskoeffizient")
output_dir.mkdir(exist_ok=True)

# 1. Read CSV file
df = pd.read_csv(input_file, encoding='utf-8')

# Spectral channels to analyze
spectral_channels = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', '555nm', 
                     '600nm', '640nm', '690nm', '745nm', '855nm', 'VIS_1', 'VIS_2', 'VIS_3']

# Create x-axis labels (numeric values only)
x_labels = ['405', '425', '450', '475', '515', '550', '555', 
            '600', '640', '690', '745', '855', 'VIS_1', 'VIS_2', 'VIS_3']

# 2. Group data by Messtyp and LED
# 3. Create named groups
groups = {
    'ref_white': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'WHITE')],
    'trans_white': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'WHITE')],
    'ref_NIR': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'NIR')],
    'trans_NIR': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'NIR')],
    #'ref_white_AGAIN=4': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 4)],
    #'ref_white_AGAIN=64': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 64)],
    #'ref_white_AGAIN=256': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 256)],
    #'ref_white_AGAIN=1024': df[(df['Messtyp'] == 'Reflexion') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 1024)],
    #'trans_white_AGAIN=4': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 4)],
    #'trans_white_AGAIN=64': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 64)],
    #'trans_white_AGAIN=256': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 256)],
    #'trans_white_AGAIN=1024': df[(df['Messtyp'] == 'Transmission') & (df['LED'] == 'WHITE') & (df['AGAIN'] == 1024)]
}

# 4. Calculate coefficient of variation
results = {}

for group_name, group_data in groups.items():
    cv_values = {}
    
    for channel in spectral_channels:
        # If group is NIR-reflection or NIR-transmission, compute only 855nm
        if group_name in ('ref_NIR', 'trans_NIR') and channel != '855nm':
            cv_values[channel] = np.nan
            continue

        # 4.1 Calculate mean per spectral channel
        mean_value = group_data[channel].mean()
        
        # 4.2 Calculate sample standard deviation
        std_value = group_data[channel].std(ddof=1)
        
        # 4.3 Calculate coefficient of variation
        if USE_MEAN_FOR_CV:
            divisor = mean_value
        else:
            divisor = CUSTOM_DIVISOR
        
        if divisor != 0:
            cv = std_value / divisor
        else:
            cv = 0  # Avoid division by zero
        
        cv_values[channel] = cv * 100  # Convert to percentage
    
    results[group_name] = cv_values

# Prepare data for plotting
df_results = pd.DataFrame(results)

# 5. Create bar chart
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(spectral_channels))
width = 0.2

# Get initial y-limits to calculate consistent text offset
temp_bars = ax.bar(x, df_results.iloc[:, 0], width)
y_min_temp, y_max_temp = ax.get_ylim()
y_range_temp = y_max_temp - y_min_temp
text_offset = 0.02 * y_range_temp  # Small offset for spacing
ax.clear()

# Plot bars for each group
for i, (group_name, color) in enumerate(zip(df_results.columns, ['blue', 'cyan', 'red', 'orange'])):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, df_results[group_name], width, label=group_name, color=color, alpha=0.8)
    
    # Add value labels on top of bars with consistent offset
    for bar in bars:
        height = bar.get_height()
        # skip NaN / non-finite values
        if not np.isfinite(height):
            continue
        ax.text(bar.get_x() + bar.get_width()/2., height + text_offset,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=8, rotation=90)

# Add padding for labels
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)

ax.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax.set_ylabel(f'Variationskoeffizient (%)', fontsize=12)
ax.set_title(f'{input_file.stem}', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Update y-axis label based on mode
if USE_MEAN_FOR_CV:
    ylabel = 'Variationskoeffizient (%)'
else:
    ylabel = f'Standardabweichung / ADCfullscale (%)'

ax.set_ylabel(ylabel, fontsize=12)

# 6. Save plots
output_filename = 'Variationskoeffizient'
png_path = output_dir / f'{output_filename}.png'
pdf_path = output_dir / f'{output_filename}.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {png_path}")

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {pdf_path}")

# Save results as CSV
csv_path = output_dir / f'{output_filename}.csv'
df_results.to_csv(csv_path, index=True)
print(f"Results saved to: {csv_path}")

#plt.show()
