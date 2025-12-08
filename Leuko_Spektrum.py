import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

# 1. Pfade definieren
# Speicherort für das PDF (angepasst an die Struktur des vorherigen Skripts)
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Analyse\Spektrum Leuko")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Daten definieren
# Die Daten werden direkt als CSV-String definiert
data = """wavelength,Al4%,L0.008,L0.016,L0.031,L0.063,L0.125,L0.25,L0.5,L1,L2,L4,L8
405,1,1,1,1,1,0.8,0.8,0.6,0.4,0.2,0.2,0
425.3,1,0.95,0.95,0.95,0.9,0.875,0.8,0.7,0.55,0.425,0.35,0.35
450.2,1,0.990909091,0.981818182,0.954545455,0.936363636,0.881818182,0.809090909,0.7,0.572727273,0.454545455,0.4,0.4
474.9,1,0.985714286,0.975,0.95,0.928571429,0.878571429,0.814285714,0.7,0.575,0.457142857,0.407142857,0.4
514.95,1,0.986904762,0.979761905,0.952380952,0.927380952,0.879761905,0.811904762,0.701190476,0.572619048,0.461904762,0.414285714,0.407142857
550.1,1,0.986774942,0.979118329,0.951972158,0.927610209,0.879582367,0.810672854,0.697911833,0.57262181,0.460556845,0.413921114,0.405568445
555.1,1,0.986928105,0.979084967,0.950980392,0.925490196,0.878431373,0.809150327,0.698039216,0.57254902,0.460130719,0.41503268,0.405882353
599.85,1,0.986871961,0.979092382,0.951863857,0.928038898,0.879416532,0.810372771,0.698541329,0.573581848,0.462236629,0.415559157,0.406807131
640.05,1,0.986798179,0.980880121,0.953566009,0.928528073,0.882094082,0.814719272,0.704097117,0.576631259,0.463732929,0.415477997,0.406373293
690.05,1,0.988416988,0.9796139,0.953667954,0.929111969,0.882316602,0.816525097,0.708571429,0.583474903,0.468571429,0.417606178,0.407876448
745.15,1,0.987967183,0.980856882,0.954056518,0.932725615,0.887876026,0.825524157,0.720510483,0.594712853,0.475478578,0.419690064,0.408751139
855.15,1,0.989981447,0.983302412,0.956586271,0.936549165,0.896474954,0.838589981,0.742857143,0.618181818,0.489053803,0.422263451,0.40890538"""

# 3. Einlesen
df = pd.read_csv(StringIO(data))

# 4. Plotten
fig, ax = plt.subplots(figsize=(14, 9))

# Zeile mit wavelength == 405 ausschließen
df_plot = df[df['wavelength'] != 405]

# Neue x-Achse: diskrete Indizes
x = range(len(df_plot))

# Alle Spalten außer 'wavelength' plotten
for column in df_plot.columns:
    if column != 'wavelength':
        ax.plot(x, df_plot[column], marker='o', label=column, linewidth=2, markersize=6)

# X-Ticks: Wellenlängen als Label, gleichmäßig verteilt
ax.set_xticks(x)
ax.set_xticklabels([f"{wl:.0f}" for wl in df_plot['wavelength']], rotation=0, fontsize=16)

# 5. Formatierung (Große Schriftarten)
ax.set_xlabel('Wellenlänge (nm)', fontsize=20, labelpad=15)
ax.set_ylabel('Normalisierte Intensität (auf Probe Al4%) (AU)', fontsize=20, labelpad=15)
ax.set_title('Leukozyten - Transmission', fontsize=26, fontweight='bold', pad=20)

# Legende außerhalb des Plots platzieren, damit der Graph nicht verdeckt wird
ax.legend(fontsize=14, title="Konzentration", title_fontsize=16, loc='center right')

# Gitter und Ticks
ax.grid(True, alpha=0.4, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()

# Speichern
pdf_path = output_dir / "Leuko_Spektrum.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path}")

# plt.show()
