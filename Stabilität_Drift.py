import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
csv_path = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Stabilität und Drift\CSV\Stabilität und Drift.csv"
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Stabilität und Drift\Ergebnis")
output_dir.mkdir(parents=True, exist_ok=True)

# Channels to process
channels = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', 
            '555nm', '600nm', '640nm', '690nm', '745nm', '855nm',
            'VIS_1', 'VIS_2', 'VIS_3']

# Read CSV
df = pd.read_csv(csv_path)

# Normalize each channel
normalized_data = {}
for channel in channels:
    max_value = df[channel].max()
    normalized_data[channel] = (df[channel] / max_value) * 100  # Convert to percentage

# Create measurement index for x-axis
measurement_index = np.arange(len(df))

# Plot 1: All channels
plt.figure(figsize=(14, 8))
for channel in channels:
    plt.plot(measurement_index, normalized_data[channel], label=channel, alpha=0.7)

plt.xlabel('Messungen (AU)', fontsize=12)
plt.ylabel('Auf Kanal Maximum normierte Intensität (%)', fontsize=12)
plt.title('Stabilität und Drift - Alle Kanäle (normiert)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save Plot 1
plt.savefig(output_dir / 'Stabilität_Drift_alle_Kanäle.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Stabilität_Drift_alle_Kanäle.pdf', bbox_inches='tight')
plt.close()

# Plot 2: Only 640nm channel
plt.figure(figsize=(12, 6))
plt.plot(measurement_index, normalized_data['640nm'], 'r-', linewidth=2, label='640nm')

plt.xlabel('Messungen (AU)', fontsize=12)
plt.ylabel('Auf Kanal Maximum normierte Intensität (%)', fontsize=12)
plt.title('Stabilität und Drift - 640nm Kanal (normiert)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save Plot 2
plt.savefig(output_dir / 'Stabilität_Drift_640nm.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Stabilität_Drift_640nm.pdf', bbox_inches='tight')
plt.close()

print(f"Plots erfolgreich gespeichert in: {output_dir}")
print(f"- Stabilität_Drift_alle_Kanäle.png/pdf")
print(f"- Stabilität_Drift_640nm.png/pdf")
