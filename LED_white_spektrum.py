import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
input_file = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Spektrometer LED Messungen\CSV\LED_Spektrum_power-variation.csv")
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Spektrometer LED Messungen\Datenverarbeitung\Spektrum LED white power variation")
output_dir.mkdir(exist_ok=True)

# 0. Read CSV file
df = pd.read_csv(input_file, sep='\t', encoding='utf-8', skiprows=1)

# Clean column names (remove whitespace)
df.columns = df.columns.str.strip()

# Extract wavelength and intensity columns
wavelength = df['&l / nm'].values
intensity_1 = df.iloc[:, 1].values  # First intensity column
intensity_2 = df.iloc[:, 2].values  # Second intensity column

# 1. & 2. Process first intensity: Add minimum and normalize
i_min_1 = np.min(intensity_1)
intensity_1_shifted = intensity_1 - i_min_1
i_max_1 = np.max(intensity_1_shifted)
intensity_1_normalized = intensity_1_shifted / i_max_1

# 1. & 2. Process second intensity: Add minimum and normalize
i_min_2 = np.min(intensity_2)
intensity_2_shifted = intensity_2 - i_min_2
i_max_2 = np.max(intensity_2_shifted)
intensity_2_normalized = intensity_2_shifted / i_max_2

# 3. Plot both spectra
plt.figure(figsize=(12, 6))
plt.plot(wavelength, intensity_1_normalized, linewidth=1.5, color='blue', label='Intensität 100%')
plt.plot(wavelength, intensity_2_normalized, linewidth=1.5, color='red', label='Intensität 20%')
plt.xlabel('Wellenlänge (nm)', fontsize=12)
plt.ylabel('Normierte Intensität (AU)', fontsize=12)
plt.title('LED weiß Spektrum - Leistungsvariation', fontsize=14)
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
output_filename = input_file.stem + '.png'
output_path = output_dir / output_filename
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Save plot as PDF
pdf_output_path = output_dir / (input_file.stem + '.pdf')
plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {pdf_output_path}")

# Also save processed data as CSV
processed_data = pd.DataFrame({
    'wavelength_nm': wavelength,
    'intensity_1_normalized': intensity_1_normalized,
    'intensity_2_normalized': intensity_2_normalized
})
csv_output_path = output_dir / (input_file.stem + '_processed.csv')
processed_data.to_csv(csv_output_path, index=False)
print(f"Processed data saved to: {csv_output_path}")

# Create additional plot showing deviation between intensities
plt.figure(figsize=(12, 6))

# Calculate relative deviation in percent only where both intensities > 10% of max
threshold = 0.1
mask = (intensity_1_normalized > threshold) & (intensity_2_normalized > threshold)

deviation_percent = np.full_like(wavelength, np.nan)
deviation_percent[mask] = ((intensity_2_normalized[mask] - intensity_1_normalized[mask]) / intensity_1_normalized[mask]) * 100

plt.plot(wavelength, deviation_percent, linewidth=1.5, color='green')
plt.xlabel('Wellenlänge (nm)', fontsize=12)
plt.ylabel('Relative Abweichung (%)', fontsize=12)
plt.title('Relative Abweichung: 20% vs 100% Intensität (nur bei >10% Intensität)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(wavelength.min(), wavelength.max())
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Set x-axis ticks in 25 nm steps
plt.xticks(np.arange(x_min, x_max + 1, 25))

plt.tight_layout()

# Save deviation plot
deviation_png = output_dir / (input_file.stem + '_deviation.png')
plt.savefig(deviation_png, dpi=300, bbox_inches='tight')
print(f"Deviation plot saved to: {deviation_png}")

deviation_pdf = output_dir / (input_file.stem + '_deviation.pdf')
plt.savefig(deviation_pdf, dpi=300, bbox_inches='tight')
print(f"Deviation plot saved to: {deviation_pdf}")

# Save deviation data
deviation_data = pd.DataFrame({
    'wavelength_nm': wavelength,
    'deviation_percent': deviation_percent
})
deviation_csv = output_dir / (input_file.stem + '_deviation.csv')
deviation_data.to_csv(deviation_csv, index=False)
print(f"Deviation data saved to: {deviation_csv}")

plt.show()
