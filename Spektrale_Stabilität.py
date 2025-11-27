import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pfad zur CSV-Datei
csv_path = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Spektrale Stabilität\CSV\Spektrale Stabilität.csv"

# Output-Verzeichnis (None = gleiches Verzeichnis wie CSV, oder eigenen Pfad angeben)
output_dir = r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Systemcharakterisierung\Datenverarbeitung\Spektrale Stabilität\Ergebnis"  # z.B. r"c:\Projekte\BA\Ergebnisse" oder None

# 1. CSV einlesen
df = pd.read_csv(csv_path)

# Spektrale Kanäle definieren
spectral_channels = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', 
                     '555nm', '600nm', '640nm', '690nm', '745nm', '855nm',
                     'VIS_1', 'VIS_2', 'VIS_3']

# 2. Mittelwert_A = Mittelwert aus den "Intensität" = 100 bilden
df_100 = df[df['Intensitaet[%]'] == 100.0]
mittelwert_A = df_100[spectral_channels].mean()

# 3. Mittelwert_B = Mittelwert aus den "Intensität" = 20 bilden
df_20 = df[df['Intensitaet[%]'] == 20.0]
mittelwert_B = df_20[spectral_channels].mean()

# 4. Mittelwert'_B = Mittelwert_B * 5
mittelwert_B_scaled = mittelwert_B * 5

# 5. Abweichung = (Mittelwert_A - Mittelwert'_B)/(Mittelwert_A + Mittelwert'_B)
abweichung = (mittelwert_A - mittelwert_B_scaled) / (mittelwert_A + mittelwert_B_scaled)

# 6. Plotte Abweichung über alle spektralen Kanäle
fig, ax = plt.subplots(figsize=(14, 7))

x_pos = np.arange(len(spectral_channels))
colors = ['blue'] * 12 + ['red'] * 3  # Spektrale Kanäle in blau, VIS in rot

# X-Achsen-Labels: nur Zahlenwerte extrahieren
x_labels = []
for ch in spectral_channels:
    if 'VIS' in ch:
        x_labels.append(ch)
    else:
        x_labels.append(ch.replace('nm', ''))

ax.bar(x_pos, abweichung.values * 100, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Wellenlänge (nm)', fontsize=12)
ax.set_ylabel('Relative Abweichung (%)', fontsize=12)
ax.set_title('Spektrale Stabilität: Relative Abweichung zwischen 100% und 20% LED weiß Intensität (skaliert)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=0, ha='center')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Berechnungsformel als Textbox hinzufügen
formula_text = (
    r'$\mathrm{Abweichung} = \frac{\overline{I}_{100\%} - 5 \cdot \overline{I}_{20\%}}{\overline{I}_{100\%} + 5 \cdot \overline{I}_{20\%}}$'
    '\n'
    r'$\overline{I}_{100\%}$: Mittelwert bei 100% Intensität'
    '\n'
    r'$\overline{I}_{20\%}$: Mittelwert bei 20% Intensität'
)
ax.text(0.98, 0.95, formula_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# 7. Speichere als PNG, PDF und CSV
if output_dir is None:
    output_dir = Path(csv_path).parent
else:
    output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)

# PNG speichern
plt.savefig(output_dir / "Spektrale_Stabilität.png", dpi=300, bbox_inches='tight')
print(f"PNG gespeichert: {output_dir / 'Spektrale_Stabilität.png'}")

# PDF speichern
plt.savefig(output_dir / "Spektrale_Stabilität.pdf", bbox_inches='tight')
print(f"PDF gespeichert: {output_dir / 'Spektrale_Stabilität.pdf'}")

plt.show()

# CSV mit Ergebnissen speichern
ergebnisse_df = pd.DataFrame({
    'Kanal': spectral_channels,
    'Mittelwert_100%': mittelwert_A.values,
    'Mittelwert_20%': mittelwert_B.values,
    'Mittelwert_20%_skaliert': mittelwert_B_scaled.values,
    'Relative_Abweichung': abweichung.values,
    'Relative_Abweichung_%': abweichung.values * 100
})

ergebnisse_df.to_csv(output_dir / "Spektrale_Stabilität_Ergebnisse.csv", 
                     index=False, sep=',', decimal='.')
print(f"CSV gespeichert: {output_dir / 'Spektrale_Stabilität_Ergebnisse.csv'}")

# Zusammenfassung ausgeben
print("\n=== Zusammenfassung ===")
print(f"Anzahl Messungen bei 100%: {len(df_100)}")
print(f"Anzahl Messungen bei 20%: {len(df_20)}")
print(f"\nMaximale Abweichung: {abweichung.abs().max()*100:.3f}% bei {abweichung.abs().idxmax()}")
print(f"Mittlere Abweichung: {abweichung.abs().mean()*100:.3f}%")
