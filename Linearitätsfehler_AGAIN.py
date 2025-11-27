import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pfade definieren
csv_file = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Linearitätsfehler AGAIN\CSV\Linearitätsfehler AGAIN.csv"

output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Linearitätsfehler AGAIN\Ergebnisse")
output_dir.mkdir(parents=True, exist_ok=True)

# CSV einlesen
df = pd.read_csv(csv_file)

# Relevante Spektralkanäle definieren
spectral_channels = ['640nm']

# Linearitätsfehler berechnen
results = []

for channel in spectral_channels:
    for idx, row in df.iterrows():
        current_again = row['AGAIN']
        measured_value = row[channel]
        
        # Digitalwert / AGAIN
        normalized_value = measured_value / current_again
        
        results.append({
            'AGAIN': current_again,
            'Channel': channel,
            'Measured': measured_value,
            'Normalized_Value': normalized_value
        })

results_df = pd.DataFrame(results)

# Plot erstellen
fig, ax = plt.subplots(figsize=(12, 8))

channel_data = results_df[results_df['Channel'] == '640nm']
x_positions = range(len(channel_data))
ax.plot(x_positions, channel_data['Normalized_Value'], 
        marker='o', label='640nm', linewidth=2, markersize=8)

ax.set_xlabel('AGAIN (AU)', fontsize=12)
ax.set_ylabel('Digitalwert / AGAIN (AU)', fontsize=12)
ax.set_title('Normalisierter Digitalwert 640nm', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(df['AGAIN'].values, rotation=45, ha='right')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()

# Ergebnisse speichern
# PNG
png_path = output_dir / "Linearitätsfehler_AGAIN.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {png_path}")

# PDF
pdf_path = output_dir / "Linearitätsfehler_AGAIN.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path}")

# CSV
csv_path = output_dir / "Linearitätsfehler_AGAIN_Ergebnisse.csv"
results_df.to_csv(csv_path, index=False)
print(f"CSV gespeichert: {csv_path}")

plt.show()
