import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.container import ErrorbarContainer

# 1. Pfade definieren
output_dir = Path(r"G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Analyse\Spektrum Leuko")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Daten definieren
# Mittelwerte (Mean)
data_mean = """wavelength,Al4%,L0.008,L0.016,L0.031,L0.063,L0.125,L0.25,L0.5,L1,L2,L4,L8
400.15,1,0.9,1,0.8,0.8,0.8,0.7,0.6,0.4,0.2,0.2,0.2
449.8,1,0.990625,0.990625,0.953125,0.934375,0.8875,0.803125,0.7,0.578125,0.45625,0.4,0.4
500.1,1,0.987165775,0.977540107,0.953475936,0.926203209,0.879679144,0.812299465,0.7,0.571657754,0.457754011,0.411229947,0.403208556
550.1,1,0.986774942,0.979118329,0.951972158,0.927610209,0.879582367,0.810672854,0.697911833,0.57262181,0.460556845,0.413921114,0.405568445
599.85,1,0.986871961,0.979092382,0.951863857,0.928038898,0.879416532,0.810372771,0.698541329,0.573581848,0.462236629,0.415559157,0.406807131
650,1,0.987843705,0.980028944,0.953111433,0.930101302,0.883212735,0.815484805,0.705209841,0.578871201,0.46512301,0.416497829,0.406946454
700.05,1,0.988057325,0.98089172,0.954617834,0.932643312,0.886783439,0.821815287,0.71433121,0.587261146,0.469745223,0.417675159,0.407643312
750.05,1,0.988910506,0.982490272,0.956225681,0.93463035,0.890856031,0.828404669,0.723346304,0.597276265,0.475875486,0.4192607,0.408171206
800.1,1,0.988927336,0.981314879,0.956401384,0.935640138,0.892733564,0.832525952,0.731487889,0.606228374,0.4816609,0.420069204,0.408304498
849.85,1,0.986363636,0.981118881,0.955944056,0.934965035,0.895104895,0.837412587,0.740909091,0.616083916,0.488111888,0.420979021,0.408391608
899.85,1,0.98962536,0.98443804,0.960230548,0.939481268,0.901440922,0.849567723,0.751008646,0.624783862,0.493371758,0.42074928,0.406916427
949.95,1,0.987671233,0.983561644,0.95890411,0.938356164,0.905479452,0.852054795,0.765753425,0.646575342,0.515068493,0.432876712,0.416438356
1000.05,1,0.979310345,0.979310345,0.937931034,0.937931034,0.896551724,0.855172414,0.772413793,0.648275862,0.482758621,0.4,0.379310345"""

# Standardabweichungen (Std)
data_std = """wavelength,Al4%,L0.008,L0.016,L0.031,L0.063,L0.125,L0.25,L0.5,L1,L2,L4,L8
170.1,0,0,0,0,0,0,0,0,0,0,0,0
170.1,0,0,0,0,0,0,0,0,0,0,0,0
340.4,0,0.07856742,0.07856742,0,0.07856742,0.07856742,0.07856742,0.07856742,0.07856742,0.07856742,0.07856742,0.07856742
350,0,0,0.070710678,0,0.070710678,0.070710678,0.070710678,0.070710678,0.070710678,0.070710678,0.070710678,0.141421356
400.15,0,0.044194174,0,0.088388348,0.088388348,0.012626907,0.031567267,0.025253814,0.03788072,0.050507627,0.050507627,0.050507627
449.8,0,0.009817291,0.009817291,0.018718302,0.018456508,0.008115628,0.006937553,0.013875105,0.025263163,0.017278433,0.001308972,0.018063816
500.1,0,0.018157296,0.002026994,0.004068262,0.015273826,0.002897745,0.014774215,0.017429291,0.02359592,0.011619528,0.002069818,0.001998445
550.1,0,0.015549144,0.00314645,0.011961743,0.016718787,0.005870633,0.013236018,0.014656031,0.024289699,0.011042471,0.000112106,6.72638E-05
599.85,0,0.015691034,0.002997215,0.011690816,0.015841189,0.006091509,0.01195166,0.0160916,0.024071622,0.011850044,0.001300139,0.001226649
650,0,0.018380935,0.002479874,0.010030756,0.01555704,0.004191193,0.011829909,0.017296173,0.025363816,0.011992404,0.000948618,0.000980824
700.05,0,0.016976033,0.003777784,0.009745723,0.016312189,0.004115039,0.014429962,0.020955104,0.026519137,0.012876991,0.000937114,0.001021095
750.05,0,0.018212858,0.002149368,0.008421294,0.015339204,0.004118643,0.014426995,0.019818033,0.027100044,0.014094215,0.000845653,0.001448572
800.1,0,0.016359125,0.007139741,0.012595563,0.018090308,0.007934899,0.010510358,0.019585168,0.028074268,0.01623772,0.002185294,0.001482812
849.85,0,0.013364561,0.006074801,0.012149601,0.020654322,0.008504721,0.008504721,0.015794481,0.024299202,0.014579521,6.93889E-18,0.00242992
899.85,0,0.016481167,0.000408624,0.011112302,0.014619658,0.001464236,0.016106595,0.016668453,0.024035036,0.013302981,0.000998859,0.003416551
949.95,0,0.019400109,0.000725238,0.007614996,0.016136539,0.000543928,0.00652714,0.015048683,0.025020701,0.016680468,0.001450475,0.006889758
1000.05,0,0.018130943,0.018130943,0.035784756,0.017176683,0.016222423,0.021947984,0.023856504,0.026719285,0.030536325,0.013836772,0.014313902"""

# 3. Einlesen
df_mean = pd.read_csv(StringIO(data_mean))
df_std = pd.read_csv(StringIO(data_std))

# Daten synchronisieren: Nur Zeilen in df_std behalten, deren Wellenlänge auch in df_mean vorkommt
df_std = df_std[df_std['wavelength'].isin(df_mean['wavelength'])]

# Sicherstellen, dass beide Dataframes nach Wellenlänge sortiert sind und der Index zurückgesetzt wird
df_mean = df_mean.sort_values('wavelength').reset_index(drop=True)
df_std = df_std.sort_values('wavelength').reset_index(drop=True)

# 4. Plotten
fig, ax = plt.subplots(figsize=(14, 9))

# Neue x-Achse: diskrete Indizes basierend auf den vorhandenen Daten
x = range(len(df_mean))

# Alle Spalten außer 'wavelength' plotten
for column in df_mean.columns:
    if column != 'wavelength':
        # Werte und Fehler abrufen
        y_values = df_mean[column]
        y_errors = df_std[column]
        
        # Plot mit Fehlerbalken
        ax.errorbar(x, y_values, yerr=y_errors, marker='o', label=column, 
                    linewidth=2, markersize=6, capsize=5, elinewidth=1.5)

# X-Ticks: Wellenlängen als Label, gleichmäßig verteilt
ax.set_xticks(x)
ax.set_xticklabels([f"{wl:.0f}" for wl in df_mean['wavelength']], rotation=0, fontsize=16)

# 5. Formatierung (Große Schriftarten)
ax.set_xlabel('Wellenlänge (nm)', fontsize=20, labelpad=15)
ax.set_ylabel('Normalisierte Intensität (auf Probe Al4%) (AU)', fontsize=20, labelpad=15)
ax.set_title('Leukozyten - Transmission', fontsize=26, fontweight='bold', pad=20)

# --- Legende anpassen (Fehlerbalken separieren) ---
handles, labels = ax.get_legend_handles_labels()
new_handles = []

# Bestehende Handles bereinigen (nur Linie/Marker, keine Fehlerbalken)
for handle in handles:
    # Farbe aus dem ErrorbarContainer extrahieren
    color = handle.lines[0].get_color()
    # Neues Handle erstellen
    new_handle = Line2D([], [], color=color, marker='o', 
                        linewidth=2, markersize=6)
    new_handles.append(new_handle)

# Dummy-Fehlerbalken für die Legende erstellen
temp_fig, temp_ax = plt.subplots()
dummy_eb = temp_ax.errorbar([0], [0], yerr=[0.5], marker='o', color='black',
                            capsize=5, capthick=1.5, linewidth=0, markersize=6,
                            elinewidth=1.5)
plt.close(temp_fig)

new_handles.append(dummy_eb)
labels.append('Variationskoeffizient')

# Legende erstellen mit HandlerMap
ax.legend(new_handles, labels, loc='center right', 
          title="Konzentration", title_fontsize=16, fontsize=14,
          handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0, yerr_size=1)})

# Gitter und Ticks
ax.grid(True, alpha=0.4, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()

# Speichern
pdf_path = output_dir / "Leuko_Spektrum.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF gespeichert: {pdf_path}")

# plt.show()
