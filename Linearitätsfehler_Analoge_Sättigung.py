import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pfade zu CSV-Dateien
base_path = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Linearitätsfehler analoge Sättigung\CSV")
output_dir = Path(r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Linearitätsfehler analoge Sättigung\Ergebnisse")
output_dir.mkdir(exist_ok=True)

csv_files = {
    'AGAIN=2': base_path / 'Linearitätsfehler Analoge Sättigung AGAIN=2.csv',
}

# Spektralen Kanal auswählen (z.B. 555nm - Peak der weißen LED)
channel = '640nm'

# Daten einlesen und verarbeiten
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

colors = {'AGAIN=2': 'blue'}
markers = {'AGAIN=2': 'o'}

results_data = []

for gain_label, csv_path in csv_files.items():
    # CSV einlesen
    df = pd.read_csv(csv_path)
    
    # Nach Intensität gruppieren und Mittelwerte berechnen
    grouped = df.groupby('Intensitaet[%]').agg({
        channel: 'mean',
        'SaturationAnalog': 'any'
    }).reset_index()
    
    intensities = grouped['Intensitaet[%]'].values
    measured = grouped[channel].values
    saturated = grouped['SaturationAnalog'].values
    
    # Normiere gemessene Werte auf Maximum
    measured_max = np.max(measured)
    measured_normalized = measured / measured_max if measured_max > 0 else measured
    
    # Erwartete lineare Werte berechnen (basierend auf ersten beiden nicht-null Punkten)
    # Finde erste beiden nicht-null Werte
    non_zero_idx = np.where(measured_normalized > 0)[0]
    if len(non_zero_idx) >= 2:
        i1, i2 = non_zero_idx[0], non_zero_idx[1]
        # Lineare Steigung berechnen
        slope = (measured_normalized[i2] - measured_normalized[i1]) / (intensities[i2] - intensities[i1])
        offset = measured_normalized[i1] - slope * intensities[i1]
        expected = slope * intensities + offset
    else:
        expected = measured_normalized.copy()
    
    # Relative Abweichung berechnen
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = np.where(expected > 0, 
                           (measured_normalized - expected) / expected * 100, 
                           0)
    
    # Plot 1: Gemessene Werte vs Erwartete lineare Werte
    ax1.plot(intensities, measured_normalized, marker=markers[gain_label], 
             color=colors[gain_label], linewidth=1.5, markersize=6,
             label='Gemessen', linestyle='-')
    ax1.plot(intensities, expected, color=colors[gain_label], 
             linewidth=1, linestyle='--', alpha=0.5,
             label='Ideal linear')
    
    # Markiere Punkte mit analoger Sättigung
    sat_mask = saturated
    if np.any(sat_mask):
        ax1.scatter(intensities[sat_mask], measured_normalized[sat_mask], 
                   s=150, facecolors='none', edgecolors='red', 
                   linewidths=2, marker='o', label='Analoge Sättigung')
    
    # Plot 2: Relative Abweichung
    ax2.plot(intensities, deviation, marker=markers[gain_label], 
             color=colors[gain_label], linewidth=1.5, markersize=6,
             label='Abweichung')
    
    # Markiere Sättigung auch in Plot 2
    if np.any(sat_mask):
        ax2.scatter(intensities[sat_mask], deviation[sat_mask], 
                   s=150, facecolors='none', edgecolors='red', 
                   linewidths=2, marker='o', label='Analoge Sättigung')
    
    # Daten für CSV speichern
    for i, intensity in enumerate(intensities):
        results_data.append({
            'AGAIN': gain_label,
            'Intensität_%': intensity,
            'Gemessen_normiert': measured_normalized[i],
            'Erwartet_Linear': expected[i],
            'Abweichung_%': deviation[i],
            'Analoge_Sättigung': saturated[i]
        })

# Plot 1 formatieren
ax1.set_xlabel('LED Intensität (%)', fontsize=12)
ax1.set_ylabel(f'Normierte Intensität {channel} (AU)', fontsize=12)
ax1.set_title(f'Linearitätsfehler bei analoger Sättigung - Kanal {channel}', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 105)

# Plot 2 formatieren
ax2.set_xlabel('LED Intensität (%)', fontsize=12)
ax2.set_ylabel('Relative Abweichung von Linearität (%)', fontsize=12)
ax2.set_title('Linearitätsabweichung', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlim(-5, 105)

# Infobox hinzufügen
info_text = (
    'Rote Kreise: Analoge Sättigung aktiv\n'
    'Gestrichelt: Ideale Linearität\n'
    'Durchgezogen: Gemessene Werte'
)
ax1.text(0.98, 0.05, info_text, transform=ax1.transAxes, 
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Speichern
plt.savefig(output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.png', 
           dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.png'}")

plt.savefig(output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.pdf', 
           bbox_inches='tight')
print(f"PDF gespeichert: {output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.pdf'}")

# CSV mit Ergebnissen speichern
results_df = pd.DataFrame(results_data)
results_df.to_csv(output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.csv', 
                 index=False)
print(f"CSV gespeichert: {output_dir / f'Linearitätsfehler_Analoge_Sättigung_{channel}.csv'}")

# Zusammenfassung ausgeben
print("\n=== Zusammenfassung ===")
for gain_label in csv_files.keys():
    gain_data = results_df[results_df['AGAIN'] == gain_label]
    sat_count = gain_data['Analoge_Sättigung'].sum()
    max_deviation = gain_data['Abweichung_%'].abs().max()
    print(f"{gain_label}:")
    print(f"  Messungen mit analoger Sättigung: {sat_count}")
    print(f"  Maximale Abweichung: {max_deviation:.2f}%")

plt.show()
