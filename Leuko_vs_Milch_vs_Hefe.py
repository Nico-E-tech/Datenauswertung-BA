import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.container import ErrorbarContainer

# 1. Pfade definieren
output_dir = Path(r"G:/Meine Ablage/Studium/6. Semester/BA Arbeit/M2 Messergebnisse 10mm/Datenauswertung/Analyse/Leuko vs Milch vs Hefe")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "Vergleich_Milch_Leuko_Hefe.csv"

# 2. CSV erstellen (Leuko_Range und Hefe_Range ergänzt)
csv_content = """Milch,Leuko,Leuko_Range,Hefe,Hefe_Range,Konzentration
0.0956,0.1032,0.015374301,,,L2
0.2708,0.2888,0.03307818,,,L1
0.4795,0.4992,0.022066874,0.4827,0.010259,L0.5; H8
0.6041,0.6840,0.01648474,0.6630,0.040882,L0.25; H4
0.7977,0.8008,0.009004576,0.7850,0.018721,L0.125; H2
0.8573,0.8816,0.022746415,0.8477,0.019854,L0.063; H1
0.9307,0.9208,0.016773089,0.9506,0.067811,L0.031; H0.5
1.0000,0.9664,0.004323941,0.9575,0.022215,L0.016; H0.25"""

with open(csv_path, "w", encoding="utf-8") as f:
    f.write(csv_content)

print(f"CSV erstellt: {csv_path}")

# 3. CSV einlesen und Plotten
df = pd.read_csv(csv_path)

# Fehlerbalken berechnen (Spannweite / 2 für symmetrische Darstellung um den Mittelwert)
df['Hefe_Yerr'] = df['Hefe_Range'] / 2
df['Leuko_Yerr'] = df['Leuko_Range'] / 2

fig, ax = plt.subplots(figsize=(12, 7))

# X-Achse Indizes
x = range(len(df))

# Plotten
# ax.plot(x, df['Milch'], marker='o', label='Milch', linewidth=2, markersize=8) # Entfernt
ax.errorbar(x, df['Leuko'], yerr=df['Leuko_Yerr'], marker='o', label='Leukozyten',
            linewidth=2, markersize=8, capsize=5, elinewidth=1.5, capthick=1.5)
ax.errorbar(x, df['Hefe'], yerr=df['Hefe_Yerr'], marker='o', label='Hefe', 
            linewidth=2, markersize=8, capsize=5, elinewidth=1.5, capthick=1.5)

# Formatierung
ax.set_xticks(x)
ax.set_xticklabels(df['Konzentration'], rotation=45, ha='right')
ax.set_ylabel('Normalisierte Intensität (auf Referenzprobe) (AU)', fontsize=16)
ax.set_xlabel('Konzentrationen', fontsize=16)
ax.set_title('Leukozyten vs. Hefezellen (600nm) - Transmission', fontsize=22, fontweight='bold')

# --- Legende anpassen (Fehlerbalken separieren) ---
handles, labels = ax.get_legend_handles_labels()
new_handles = []

for handle in handles:
    if isinstance(handle, ErrorbarContainer):
        color = handle.lines[0].get_color()
    else:
        color = handle.get_color()
    new_handle = Line2D([], [], color=color, marker='o', 
                        linewidth=2, markersize=8)
    new_handles.append(new_handle)

# Dummy-Fehlerbalken für die Legende erstellen
temp_fig, temp_ax = plt.subplots()
dummy_eb = temp_ax.errorbar([0], [0], yerr=[0.5], marker='o', color='black',
                            capsize=5, capthick=1.5, linewidth=0, markersize=8,
                            elinewidth=1.5)
plt.close(temp_fig)

new_handles.append(dummy_eb)
labels.append('Spannweite min-max')

ax.legend(new_handles, labels, loc='lower right', fontsize=16,
          handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0, yerr_size=1)})

ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()

# Speichern
pdf_path = output_dir / "Leuko_vs_Milch_vs_Hefe.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path}")

#plt.show()
