import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pfade definieren
csv_file = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Temperatur - Kreuzempfindlichkeit LED\CSV\Kreuzempfindlichkeit LED Temperatur weiß.csv"
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Temperatur - Kreuzempfindlichkeit LED\Ergebnisse")
output_dir.mkdir(parents=True, exist_ok=True)

# CSV einlesen
df = pd.read_csv(csv_file)

# Temperaturen definieren
T1 = 24.8  # °C
T2 = 55.4  # °C
delta_T = T2 - T1

# Spektralkanäle definieren
spectral_channels = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', 
                     '555nm', '600nm', '640nm', '690nm', '745nm', '855nm',
                     'VIS_1', 'VIS_2', 'VIS_3']

# Daten für T1 und T2 extrahieren
df_T1 = df[df['Text1'] == 'T1']
df_T2 = df[df['Text1'] == 'T2']

# ===== PLOT 1: Absolute Spektralwerte =====
fig1, ax1 = plt.subplots(figsize=(14, 8))

x_positions = range(len(spectral_channels))
values_T1 = [df_T1[ch].values[0] for ch in spectral_channels]
values_T2 = [df_T2[ch].values[0] for ch in spectral_channels]

width = 0.35
bars1 = ax1.bar([x - width/2 for x in x_positions], values_T1, width, 
        label=f'T1 = {T1}°C', alpha=0.8)
bars2 = ax1.bar([x + width/2 for x in x_positions], values_T2, width, 
        label=f'T2 = {T2}°C', alpha=0.8)

# Datenbeschriftungen hinzufügen
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, rotation=90)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, rotation=90)

# X-Achsen Labels vorbereiten (nur Zahlen)
x_labels = []
for ch in spectral_channels:
    if ch.startswith('VIS_'):
        x_labels.append(ch)
    else:
        x_labels.append(ch.replace('nm', ''))

ax1.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax1.set_ylabel('Digitalwert (Counts)', fontsize=12)
ax1.set_title('Spektrale Kanäle bei verschiedenen Temperaturen', fontsize=14, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=0, ha='right')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(top=max(max(values_T1), max(values_T2)) * 1.15)

plt.tight_layout()

# Plot 1 speichern
png_path1 = output_dir / "Spektralwerte_Temperaturvergleich.png"
plt.savefig(png_path1, dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {png_path1}")

pdf_path1 = output_dir / "Spektralwerte_Temperaturvergleich.pdf"
plt.savefig(pdf_path1, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path1}")

# CSV für Plot 1
df_plot1 = pd.DataFrame({
    'Channel': spectral_channels,
    f'T1_{T1}C': values_T1,
    f'T2_{T2}C': values_T2
})
csv_path1 = output_dir / "Spektralwerte_Temperaturvergleich.csv"
df_plot1.to_csv(csv_path1, index=False)
print(f"CSV gespeichert: {csv_path1}")

# ===== PLOT 2: Temperatur-Kreuzempfindlichkeit K_Trel =====
K_Trel_results = []

for channel in spectral_channels:
    S_T1 = df_T1[channel].values[0]
    S_T2 = df_T2[channel].values[0]
    
    # K_Trel [%/K] = ((S_T2 - S_T1) / S_T1) / delta_T * 100
    if S_T1 != 0:
        K_Trel = ((S_T2 - S_T1) / S_T1) / delta_T * 100
    else:
        K_Trel = 0
    
    K_Trel_results.append({
        'Channel': channel,
        'K_Trel_%/K': K_Trel,
        'S_T1': S_T1,
        'S_T2': S_T2
    })

df_K_Trel = pd.DataFrame(K_Trel_results)

fig2, ax2 = plt.subplots(figsize=(14, 8))

bars = ax2.bar(x_positions, df_K_Trel['K_Trel_%/K'], alpha=0.8, color='steelblue')

# Datenbeschriftungen hinzufügen
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

ax2.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax2.set_ylabel('K_Trel [%/K]', fontsize=12)
ax2.set_title('Temperatur-Kreuzempfindlichkeit K_Trel', fontsize=14, fontweight='bold')
ax2.set_xticks(x_positions)
ax2.set_xticklabels(x_labels, rotation=0, ha='right')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Plot 2 speichern
png_path2 = output_dir / "Temperatur_Kreuzempfindlichkeit.png"
plt.savefig(png_path2, dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {png_path2}")

pdf_path2 = output_dir / "Temperatur_Kreuzempfindlichkeit.pdf"
plt.savefig(pdf_path2, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path2}")

# CSV für Plot 2
csv_path2 = output_dir / "Temperatur_Kreuzempfindlichkeit.csv"
df_K_Trel.to_csv(csv_path2, index=False)
print(f"CSV gespeichert: {csv_path2}")

# ===== NIR LED DATEN =====
csv_file_nir = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Temperatur - Kreuzempfindlichkeit LED\CSV\Kreuzempfindlichkeit LED Temperatur NIR.csv"
df_nir = pd.read_csv(csv_file_nir)

# Daten für T1 und T2 extrahieren (NIR)
df_T1_nir = df_nir[df_nir['Text1'] == 'T1']
df_T2_nir = df_nir[df_nir['Text1'] == 'T2']

# ===== PLOT 3: Absolute Spektralwerte NIR =====
fig3, ax3 = plt.subplots(figsize=(14, 8))

values_T1_nir = [df_T1_nir[ch].values[0] for ch in spectral_channels]
values_T2_nir = [df_T2_nir[ch].values[0] for ch in spectral_channels]

bars1_nir = ax3.bar([x - width/2 for x in x_positions], values_T1_nir, width, 
        label=f'T1 = {T1}°C', alpha=0.8)
bars2_nir = ax3.bar([x + width/2 for x in x_positions], values_T2_nir, width, 
        label=f'T2 = {T2}°C', alpha=0.8)

# Datenbeschriftungen hinzufügen
for bar in bars1_nir:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, rotation=90)
for bar in bars2_nir:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, rotation=90)

ax3.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax3.set_ylabel('Digitalwert (Counts)', fontsize=12)
ax3.set_title('Spektrale Kanäle bei verschiedenen Temperaturen (NIR LED)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_positions)
ax3.set_xticklabels(x_labels, rotation=0, ha='right')
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(top=max(max(values_T1_nir), max(values_T2_nir)) * 1.15)

plt.tight_layout()

# Plot 3 speichern
png_path3 = output_dir / "Spektralwerte_Temperaturvergleich_NIR.png"
plt.savefig(png_path3, dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {png_path3}")

pdf_path3 = output_dir / "Spektralwerte_Temperaturvergleich_NIR.pdf"
plt.savefig(pdf_path3, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path3}")

# CSV für Plot 3
df_plot3 = pd.DataFrame({
    'Channel': spectral_channels,
    f'T1_{T1}C': values_T1_nir,
    f'T2_{T2}C': values_T2_nir
})
csv_path3 = output_dir / "Spektralwerte_Temperaturvergleich_NIR.csv"
df_plot3.to_csv(csv_path3, index=False)
print(f"CSV gespeichert: {csv_path3}")

# ===== PLOT 4: Temperatur-Kreuzempfindlichkeit K_Trel für 855nm (NIR) =====
channel_855 = '855nm'
S_T1_855 = df_T1_nir[channel_855].values[0]
S_T2_855 = df_T2_nir[channel_855].values[0]

if S_T1_855 != 0:
    K_Trel_855 = ((S_T2_855 - S_T1_855) / S_T1_855) / delta_T * 100
else:
    K_Trel_855 = 0

fig4, ax4 = plt.subplots(figsize=(8, 8))

bar_855 = ax4.bar(0, K_Trel_855, alpha=0.8, color='steelblue', width=0.5)

# Datenbeschriftung hinzufügen
ax4.text(0, K_Trel_855, f'{K_Trel_855:.3f}', 
         ha='center', va='bottom' if K_Trel_855 >= 0 else 'top', fontsize=12)

ax4.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax4.set_ylabel('K_Trel [%/K]', fontsize=12)
ax4.set_title('Temperatur-Kreuzempfindlichkeit K_Trel bei 855nm (NIR LED)', fontsize=14, fontweight='bold')
ax4.set_xticks([0])
ax4.set_xticklabels(['855'])
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Plot 4 speichern
png_path4 = output_dir / "Temperatur_Kreuzempfindlichkeit_855nm_NIR.png"
plt.savefig(png_path4, dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {png_path4}")

pdf_path4 = output_dir / "Temperatur_Kreuzempfindlichkeit_855nm_NIR.pdf"
plt.savefig(pdf_path4, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path4}")

# CSV für Plot 4
df_plot4 = pd.DataFrame({
    'Channel': [channel_855],
    'K_Trel_%/K': [K_Trel_855],
    'S_T1': [S_T1_855],
    'S_T2': [S_T2_855]
})
csv_path4 = output_dir / "Temperatur_Kreuzempfindlichkeit_855nm_NIR.csv"
df_plot4.to_csv(csv_path4, index=False)
print(f"CSV gespeichert: {csv_path4}")

#plt.show()
