import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Pfade definieren
output_dir = Path(r"G:/Meine Ablage/Studium/6. Semester/BA Arbeit/M2 Messergebnisse 10mm/Datenauswertung/Analyse/Leuko vs Milch vs Hefe")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "Vergleich_Milch_Leuko_Hefe.csv"

# 2. CSV erstellen (Daten bereinigt um %)
# Die Daten werden direkt als String definiert und in eine Datei geschrieben
csv_content = """Milch,Leuko,Hefe,Konzentration
0.0956,0.1032,,L2
0.2708,0.2888,,L1
0.4795,0.4992,0.4827,L0.5; H8
0.6041,0.6840,0.6630,L0.25; H4
0.7977,0.8008,0.7850,L0.125; H2
0.8573,0.8816,0.8477,L0.063; H1
0.9307,0.9208,0.9506,L0.031; H0.5
1.0000,0.9664,0.9575,L0.016; H0.25"""

with open(csv_path, "w", encoding="utf-8") as f:
    f.write(csv_content)

print(f"CSV erstellt: {csv_path}")

# 3. CSV einlesen und Plotten
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(12, 7))

# X-Achse Indizes
x = range(len(df))

# Plotten
ax.plot(x, df['Milch'], marker='o', label='Milch', linewidth=2, markersize=8)
ax.plot(x, df['Leuko'], marker='o', label='Leukozyten', linewidth=2, markersize=8)
ax.plot(x, df['Hefe'], marker='o', label='Hefe', linewidth=2, markersize=8)

# Formatierung
ax.set_xticks(x)
ax.set_xticklabels(df['Konzentration'], rotation=45, ha='right')
ax.set_ylabel('Normalisierte Intensität (auf Referenzprobe) (AU)', fontsize=16)
ax.set_xlabel('Konzentrationen', fontsize=16)
ax.set_title('Vergleich: Milch vs. Leuko vs. Hefe (600nm) - Transmission', fontsize=22, fontweight='bold')
ax.legend(fontsize=16)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()

# Speichern
pdf_path = output_dir / "Leuko_vs_Milch_vs_Hefe.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path}")

#plt.show()
