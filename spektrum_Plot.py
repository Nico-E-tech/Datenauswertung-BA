import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Einstellbare Referenzprobe
REFERENCE_PROBE = 'I'  # Ändere diesen Wert um eine andere Probe als Referenz zu verwenden
SAVE_VECTOR_FORMAT = True  # True für zusätzliche Vektorgrafik (PDF), False nur PNG
VECTOR_FORMAT = 'pdf'  # 'pdf' oder 'svg'

# Datei einlesen
file_path = r'G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 1mm\Datenauswertung\Hefe\CSV\Messreihe Hefe Lena Messvorschrift C.csv'
df = pd.read_csv(file_path)

# Dateinamen ohne Pfad und Endung extrahieren
base_filename = os.path.splitext(os.path.basename(file_path))[0]
output_dir = r'G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 1mm\Datenauswertung\Hefe\Spektrum'

# Wellenlängen aus den Spaltennamen extrahieren (405nm bis 855nm)
wavelength_columns = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', 
                      '555nm', '600nm', '640nm', '690nm', '745nm', '855nm']
wavelengths = [int(col.replace('nm', '')) for col in wavelength_columns]

# Funktion zum Normieren auf die Referenzprobe
def normalize_to_reference_probe(data_subset, wavelength_cols, reference_probe=REFERENCE_PROBE):
    """Normiert alle Spektralwerte auf die gewählte Referenzprobe (Text1)"""
    probe_ref = data_subset[data_subset['Text1'] == reference_probe]
    
    if len(probe_ref) == 0:
        print(f"Warnung: Probe {reference_probe} nicht gefunden für {data_subset['Messtyp'].iloc[0]}, {data_subset['LED'].iloc[0]}")
        return data_subset
    
    # Referenzwerte von der gewählten Probe
    ref_values = probe_ref[wavelength_cols].values[0]
    
    # Normierung durchführen
    normalized_data = data_subset.copy()
    for col in wavelength_cols:
        col_idx = wavelength_cols.index(col)
        normalized_data[col] = data_subset[col] / ref_values[col_idx]
    
    return normalized_data

# Funktion zum Plotten
def create_plot(messtyp, led, ax):
    """Erstellt einen Plot für die gegebene Kombination von Messtyp und LED"""
    # Daten filtern
    filtered = df[(df['Messtyp'] == messtyp) & (df['LED'] == led)].copy()
    
    if len(filtered) == 0:
        ax.text(0.5, 0.5, f'Keine Daten für\n{messtyp}, {led}', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Normierung auf Referenzprobe
    normalized = normalize_to_reference_probe(filtered, wavelength_columns)
    
    # Plot für jede Probe erstellen
    for idx, row in normalized.iterrows():
        spectrum_values = [row[col] for col in wavelength_columns]
        label = row['Text2']  # Text2 als Legende verwenden
        ax.plot(range(len(wavelengths)), spectrum_values, marker='o', label=label, linewidth=2)
    
    # Plot formatieren
    ax.set_xlabel('Wellenlänge [nm]', fontsize=10)
    ax.set_ylabel(f'Normierte Intensität (rel. zu Probe {REFERENCE_PROBE} = Wasser)', fontsize=10)
    
    # Titel anpassen basierend auf Dateiname
    if base_filename.endswith('_merged'):
        title = f'{messtyp}'
    else:
        title = f'{messtyp} - {led}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.legend(title='Konzentration', fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(wavelengths)))
    ax.set_xticklabels(wavelengths)
    ax.tick_params(axis='x', rotation=0)

# Vier separate Plots erstellen
combinations = [
    ('Reflexion', 'WHITE', 'ref_white'),
    #('Reflexion', 'NIR', 'ref_NIR'),
    ('Transmission', 'WHITE', 'trans_white'),
    #('Transmission', 'NIR', 'trans_NIR')
]

for messtyp, led, suffix in combinations:
    fig, ax = plt.subplots(figsize=(10, 6))
    create_plot(messtyp, led, ax)
    plt.tight_layout()
    
    # PNG speichern
    output_filename = f"{base_filename}_{suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot gespeichert: {output_path}")
    
    # Vektorgrafik speichern (optional)
    if SAVE_VECTOR_FORMAT:
        vector_filename = f"{base_filename}_{suffix}.{VECTOR_FORMAT}"
        vector_path = os.path.join(output_dir, vector_filename)
        plt.savefig(vector_path, format=VECTOR_FORMAT, bbox_inches='tight')
        print(f"Vektorgrafik gespeichert: {vector_path}")
    
    #plt.show()

print("Alle 4 Plots wurden erstellt, angezeigt und gespeichert.")
