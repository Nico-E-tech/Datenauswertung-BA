import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
input_file = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Spektrum weiße LED\CSV\LED_Spektrum_Spektrometer.csv")
as7343_file = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Spektrum weiße LED\CSV\Messvorschrift A.csv")
output_dir = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Spektrum weiße LED\Datenverarbeitung\Spektrometer vs AS7343")
output_dir.mkdir(exist_ok=True)

# 0. Read CSV file
df = pd.read_csv(input_file, sep='\t', encoding='utf-8', skiprows=1)

# Clean column names (remove whitespace)
df.columns = df.columns.str.strip()

# Debug: Print available columns
print("Available columns:", df.columns.tolist())
print(f"Number of columns: {len(df.columns)}")
print(f"Number of rows: {len(df)}")

# Extract wavelength and intensity columns - use first column for wavelength
wavelength = df.iloc[:, 0].values

# Check if we have enough columns
if len(df.columns) >= 2:
    intensity_1 = df.iloc[:, 1].values  # First intensity column (100%)
else:
    raise ValueError(f"Not enough columns in CSV file. Expected at least 2, found {len(df.columns)}")

# 1. & 2. Process first intensity: Add minimum and normalize
i_min_1 = np.min(intensity_1)
intensity_1_shifted = intensity_1 - i_min_1
i_max_1 = np.max(intensity_1_shifted)
intensity_1_normalized = intensity_1_shifted / i_max_1

# 1. Read AS7343 CSV file
df_as7343 = pd.read_csv(as7343_file)

# Define spectral channels and their center wavelengths
spectral_channels = {
    '405nm': 405, '425nm': 425, '450nm': 450, '475nm': 475,
    '515nm': 515, '550nm': 550, '555nm': 555, '600nm': 600,
    '640nm': 640, '690nm': 690, '745nm': 745, '855nm': 855
}

# Extract intensity values for AS7343
as7343_wavelengths = []
as7343_intensities = []
for channel, wl in spectral_channels.items():
    as7343_wavelengths.append(wl)
    as7343_intensities.append(df_as7343[channel].values[0])

as7343_wavelengths = np.array(as7343_wavelengths)
as7343_intensities = np.array(as7343_intensities)

# 2. Normalize AS7343 data
as7343_min = np.min(as7343_intensities)
as7343_shifted = as7343_intensities - as7343_min
as7343_max = np.max(as7343_shifted)
as7343_normalized = as7343_shifted / as7343_max

# 3. Plot both spectra together
plt.figure(figsize=(12, 6))
plt.plot(wavelength, intensity_1_normalized, linewidth=1.5, color='blue', label='Spektrometer')
plt.plot(as7343_wavelengths, as7343_normalized, linewidth=1.5, color='green', 
         linestyle='-', marker='o', markersize=8, markeredgecolor='black', 
         markeredgewidth=1.5, label='AS7343', zorder=5)
plt.xlabel('Wellenlänge (nm)', fontsize=12)
plt.ylabel('Normierte Intensität (AU)', fontsize=12)
plt.title('Spektrometer vs AS7343 - LED weiß', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(wavelength.min(), wavelength.max())
plt.ylim(0, 1.05)

# Set x-axis ticks in 25 nm steps
x_min = int(wavelength.min() / 25) * 25
x_max = int(wavelength.max() / 25) * 25 + 25
plt.xticks(np.arange(x_min, x_max + 1, 25))

plt.tight_layout()

# 4. Save plot with same name as CSV
output_filename = input_file.stem + '_comparison.png'
output_path = output_dir / output_filename
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Save plot as PDF
pdf_output_path = output_dir / (input_file.stem + '_comparison.pdf')
plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {pdf_output_path}")

# Also save processed data as CSV
processed_data = pd.DataFrame({
    'wavelength_nm': wavelength,
    'intensity_normalized': intensity_1_normalized
})
csv_output_path = output_dir / (input_file.stem + '_processed.csv')
processed_data.to_csv(csv_output_path, index=False)
print(f"Processed data saved to: {csv_output_path}")

# Save AS7343 data
as7343_data = pd.DataFrame({
    'wavelength_nm': as7343_wavelengths,
    'intensity_normalized': as7343_normalized
})
as7343_csv = output_dir / (input_file.stem + '_AS7343.csv')
as7343_data.to_csv(as7343_csv, index=False)
print(f"AS7343 data saved to: {as7343_csv}")

plt.show()
