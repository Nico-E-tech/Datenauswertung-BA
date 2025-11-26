import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re
import os

# Configuration
CSV_FILE = r'G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Milch\CSV\Messreihe Milch.csv'
REFERENCE_PROBE = 'I'  # Probe zur Normierung (0 µg/mL Blank)
WAVELENGTH = '855nm'  # Wellenlänge für die Analyse
OUTPUT_DIR = r'G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Milch\Intensität vs Konzentration'
SAMPLE_THICKNESS_MM = 10  # Schichtdicke in mm

# Regression-Einstellungen
USE_EXPONENTIAL = True   # Exponentielle Regression aktivieren
USE_POWER_LAW = True     # Power Law Regression aktivieren
USE_LINEAR = False        # Lineare Regression aktivieren

# Kombinationen für die Graphen
combinations = [
    #('Reflexion', 'WHITE', 'ref_white'),
    ('Reflexion', 'NIR', 'ref_NIR'),
    #('Transmission', 'WHITE', 'trans_white'),
    ('Transmission', 'NIR', 'trans_NIR'),
]

def read_and_normalize_data(csv_file, reference_probe):
    """
    Liest CSV-Datei ein und normiert ALLE Wellenlängen auf Referenzprobe
    """
    # CSV einlesen
    df = pd.read_csv(csv_file)
    
    # Erstelle normierte Kopie der Daten
    df_normalized = df.copy()
    
    # Finde alle Wellenlängen-Spalten (numerische Spalten mit 'nm' im Namen)
    wavelength_columns = [col for col in df.columns if 'nm' in col and col[0].isdigit()]
    
    # Finde Referenzwerte für jede Messtyp/LED Kombination
    for messtyp in df['Messtyp'].unique():
        for led in df['LED'].unique():
            # Referenzwert für diese Kombination
            ref_mask = (df['Text1'] == reference_probe) & \
                       (df['Messtyp'] == messtyp) & \
                       (df['LED'] == led)
            
            if ref_mask.any():
                # Normiere ALLE Wellenlängen-Spalten
                comb_mask = (df['Messtyp'] == messtyp) & (df['LED'] == led)
                
                for wl_col in wavelength_columns:
                    ref_value = df.loc[ref_mask, wl_col].values[0]
                    
                    if ref_value != 0:
                        df_normalized.loc[comb_mask, wl_col] = \
                            df.loc[comb_mask, wl_col] / ref_value
    
    return df, df_normalized

def extract_concentration(text):
    """
    Extrahiert numerischen Konzentrationswert aus Text2
    """
    match = re.search(r'(\d+\.?\d*)', str(text))
    if match:
        return float(match.group(1))
    return 0.0

def fit_models(X, y):
    """
    Testet verschiedene Regressionsmodelle und gibt das beste zurück
    """
    models = {}
    
    # Prüfe ob Daten für exponentielle Absorptions-Regression geeignet sind
    if len(X) > 1 and np.all(y > 0):
        # Berechne Korrelation zwischen Konzentration und Intensität
        correlation = np.corrcoef(X.flatten(), y)[0, 1]
        
        # Debug: Zeige Daten
        print(f"  Datenpunkte: {len(X)}")
        print(f"  Korrelation: {correlation:.4f}")
        
        # Für Absorption erwarten wir NEGATIVE Korrelation:
        # Höhere Konzentration → Niedrigere Intensität
        if correlation < -0.2:  # Negative Korrelation (fallende Kurve)
            
            # Exponentielle Regression
            if USE_EXPONENTIAL:
                y_log = np.log(y)
                exp_model = LinearRegression()
                exp_model.fit(X, y_log)
                
                # Berechne Vorhersagen für alle Datenpunkte
                y_pred_log = exp_model.predict(X)
                y_pred_exp = np.exp(y_pred_log)
                r2_exp = r2_score(y, y_pred_exp)
                
                # Berechne Absorptionskoeffizient µa
                # Modell: I/I0 = e^(-µa * c * d)
                # ln(I/I0) = -µa * c * d
                # slope = -µa * d -> µa = -slope / d
                slope = exp_model.coef_[0]
                mu_a = -slope / SAMPLE_THICKNESS_MM  # [1/(µg/mL * mm)]
                intercept = exp_model.intercept_
                
                models['Exponential'] = {
                    'model': exp_model,
                    'y_pred': y_pred_exp,
                    'r2': r2_exp,
                    'poly_features': None,
                    'is_exp': True,
                    'mu_a': mu_a,
                    'slope': slope,
                    'intercept': intercept
                }
                
                print(f"  ✓ Exponentielle Regression erfolgreich")
                print(f"  ✓ R² = {r2_exp:.4f}")
            
            # Power Law Regression: y = a * x^b
            if USE_POWER_LAW:
                # Log-Transformation: ln(y) = ln(a) + b * ln(x)
                # Filtere Nullwerte für Log-Transformation
                X_nonzero = X[X > 0]
                y_nonzero = y[X.flatten() > 0]
                
                if len(X_nonzero) >= 2:
                    X_log = np.log(X_nonzero)
                    y_log_power = np.log(y_nonzero)
                    
                    power_model = LinearRegression()
                    power_model.fit(X_log.reshape(-1, 1), y_log_power)
                    
                    # Vorhersage auf allen Datenpunkten (außer x=0)
                    y_pred_log_power = power_model.predict(np.log(X[X > 0]).reshape(-1, 1))
                    y_pred_power = np.exp(y_pred_log_power)
                    r2_power = r2_score(y[X.flatten() > 0], y_pred_power)
                    
                    # Parameter: y = a * x^b
                    b_power = power_model.coef_[0]
                    ln_a_power = power_model.intercept_
                    a_power = np.exp(ln_a_power)
                    
                    models['Power Law'] = {
                        'model': power_model,
                        'y_pred': y_pred_power,
                        'r2': r2_power,
                        'is_power': True,
                        'a': a_power,
                        'b': b_power
                    }
                    
                    print(f"  ✓ Power Law Regression erfolgreich")
                    print(f"  ✓ R² = {r2_power:.4f}")
            
            # Lineare Regression: y = m * x + b
            if USE_LINEAR:
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                
                y_pred_linear = linear_model.predict(X)
                r2_linear = r2_score(y, y_pred_linear)
                
                # Parameter: y = m * x + b
                m_linear = linear_model.coef_[0]
                b_linear = linear_model.intercept_
                
                models['Linear'] = {
                    'model': linear_model,
                    'y_pred': y_pred_linear,
                    'r2': r2_linear,
                    'is_linear': True,
                    'm': m_linear,
                    'b': b_linear
                }
                
                print(f"  ✓ Lineare Regression erfolgreich")
                print(f"  ✓ R² = {r2_linear:.4f}")
            
        elif correlation > 0.2:  # Positive Korrelation (steigende Kurve)
            print(f"  ✗ WARNUNG: Positive Korrelation ({correlation:.3f})")
            print(f"    Intensität STEIGT mit Konzentration → keine Absorption!")
        else:  # Schwache oder keine Korrelation
            print(f"  ⚠ Schwache Korrelation ({correlation:.3f})")
            print(f"    Kein klarer exponentieller Trend")
    
    # Finde bestes Modell
    if models:
        best_name = max(models, key=lambda k: models[k]['r2'])
        best_model = models[best_name]
    else:
        best_name = None
        best_model = None
    
    return models, best_name, best_model

def create_graphs(df_normalized, messtyp, led, wavelength, label):
    """
    Erstellt beide Graphen für eine Kombination
    """
    # Filtere Daten nach Messtyp und LED
    mask = (df_normalized['Messtyp'] == messtyp) & \
           (df_normalized['LED'] == led)
    
    data = df_normalized[mask].copy()
    
    if len(data) == 0:
        print(f"  FEHLER: Keine Daten gefunden für {messtyp} - {led}")
        return None, None, 0.0, 'None'
    
    print(f"  Gefunden: {len(data)} Datenpunkte für {wavelength}")
    
    # Sortiere nach Text1 (Probe)
    data = data.sort_values('Text1')
    
    # Extrahiere Konzentrationen
    data['Concentration'] = data['Text2'].apply(extract_concentration)
    
    # Sortiere nach Konzentration für bessere Darstellung
    data = data.sort_values('Concentration')
    
    # Print normierte Werte für ausgewählte Wellenlänge
    print(f"\n  Normierte Werte für {wavelength}:")
    for idx, row in data.iterrows():
        print(f"    {row['Text1']} ({row['Concentration']:.2f} µg/mL): {row[wavelength]:.4f}")
    print()
    
    # Graph 1: Intensität vs Konzentration (numerisch)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(data['Concentration'], data[wavelength], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Konzentration [µg/mL]', fontsize=12)
    ax1.set_ylabel(f'Normiert auf Probe {REFERENCE_PROBE}', fontsize=12)
    ax1.set_title(f'{messtyp} - {led} - {wavelength}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Graph 2: Vergleich mehrerer Regressionsmodelle
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    X = data['Concentration'].values.reshape(-1, 1)
    y = data[wavelength].values
    
    # Fitte verschiedene Modelle
    models, best_name, best_model = fit_models(X, y)
    
    if not models:
        # Kein Modell verfügbar
        ax2.scatter(X, y, s=100, alpha=0.7, color='black', label='Messwerte')
        
        # Prüfe Korrelation für detaillierte Warnung
        if len(X) > 1:
            correlation = np.corrcoef(X.flatten(), y)[0, 1]
            
            if correlation > 0.3:
                warning_text = f'Korrelation: {correlation:.3f}\n\n'
                warning_text += '⚠ POSITIVE KORRELATION ⚠\n\n'
                warning_text += 'Intensität STEIGT mit Konzentration!\n'
                warning_text += 'Dies deutet NICHT auf Absorption hin.\n\n'
                warning_text += 'Mögliche Ursachen:\n'
                warning_text += '• Streueffekte\n'
                warning_text += '• Fluoreszenz\n'
                warning_text += '• Falsche Wellenlänge\n\n'
                warning_text += '→ Andere Wellenlänge wählen!'
                color = 'orange'
            elif abs(correlation) < 0.3:
                warning_text = f'Korrelation: {correlation:.3f}\n\n'
                warning_text += 'Schwache oder keine Korrelation\n'
                warning_text += 'Kein klarer Trend erkennbar'
                color = 'yellow'
            else:
                warning_text = 'Zu wenig Datenpunkte\n'
                warning_text += 'für Regression'
                color = 'yellow'
        else:
            warning_text = 'Zu wenig Datenpunkte\nfür Regression'
            color = 'yellow'
        
        ax2.text(0.5, 0.5, warning_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        ax2.set_xlabel('Konzentration [µg/mL]', fontsize=12)
        ax2.set_ylabel(f'Normiert auf Probe {REFERENCE_PROBE}', fontsize=12)
        ax2.set_title(f'{messtyp} - {led} - {wavelength} - Keine Regression möglich', 
                     fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig1, fig2, 0.0, 'None'
    
    # Plotte alle Datenpunkte
    ax2.scatter(X, y, s=100, alpha=0.7, color='black', label='Messwerte', zorder=5)
    
    # Erstelle glatte Kurve für Visualisierung (von 0 bis X_max)
    X_min = 0 if np.any(X == 0) else X.min()
    X_max = X.max()
    X_smooth = np.linspace(X_min, X_max, 200).reshape(-1, 1)
    
    colors = ['red', 'blue', 'green', 'orange']
    formula_texts = []
    
    for i, (name, model_info) in enumerate(models.items()):
        if 'is_exp' in model_info:
            # Exponentielles Modell
            y_smooth_log = model_info['model'].predict(X_smooth)
            y_smooth = np.exp(y_smooth_log)
            
            # Erstelle Formel-Text mit hochgestelltem Exponenten
            mu_a = model_info['mu_a']
            formula_text = r'Exponential:' + '\n'
            formula_text += r'$\frac{I}{I_0} = e^{-\mu_a \cdot c \cdot d}$' + '\n'
            formula_text += f'$\\mu_a$ = {mu_a:.4f} (µg/mL·mm)$^{{-1}}$\n'
            formula_text += f'd = {SAMPLE_THICKNESS_MM} mm'
            formula_texts.append(formula_text)
            
        elif 'is_power' in model_info:
            # Power Law Modell: y = a * x^b
            # Nur für x > 0 plotten
            X_smooth_nonzero = X_smooth[X_smooth > 0]
            y_smooth_log = model_info['model'].predict(np.log(X_smooth_nonzero).reshape(-1, 1))
            y_smooth = np.exp(y_smooth_log)
            
            # Erstelle Formel-Text im gleichen Stil
            a_power = model_info['a']
            b_power = model_info['b']
            formula_text = r'Power Law:' + '\n'
            formula_text += r'$\frac{I}{I_0} = a \cdot c^b$' + '\n'
            formula_text += f'$a$ = {a_power:.4f}\n'
            formula_text += f'$b$ = {b_power:.4f}\n'
            formula_texts.append(formula_text)
            # Verwende nicht-zero X für Power Law Plot
            X_smooth = X_smooth_nonzero
            
        elif 'is_linear' in model_info:
            # Lineares Modell: y = m * x + b
            y_smooth = model_info['model'].predict(X_smooth)
            
            # Erstelle Formel-Text
            m_linear = model_info['m']
            b_linear = model_info['b']
            formula_text = r'Linear:' + '\n'
            formula_text += r'$\frac{I}{I_0} = m \cdot c + b$' + '\n'
            formula_text += f'$m$ = {m_linear:.6f}\n'
            formula_text += f'$b$ = {b_linear:.4f}\n'
            formula_texts.append(formula_text)
        
        linestyle = '-'
        linewidth = 2.5
        alpha = 0.9
        
        # Stern hinzufügen für bestes Modell
        star = ' ★' if name == best_name else ''
        label_text = f'{name}: R² = {model_info["r2"]:.4f}{star}'
        
        ax2.plot(X_smooth, y_smooth, linestyle, linewidth=linewidth, 
                alpha=alpha, color=colors[i % len(colors)], label=label_text)
    
    # Füge Formel-Texte zum Graphen hinzu (mittig rechts)
    if formula_texts:
        combined_formula = '\n\n'.join(formula_texts)
        ax2.text(0.98, 0.50, combined_formula, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Konzentration [µg/mL]', fontsize=12)
    ax2.set_ylabel(f'Normiert auf Probe {REFERENCE_PROBE}', fontsize=12)
    
    # Titel anpassen je nach bestem Modell
    if best_name:
        title_suffix = best_name
    else:
        title_suffix = 'Regression'
    
    ax2.set_title(f'{messtyp} - {led} - {wavelength} - {title_suffix}', 
                 fontsize=14)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig1, fig2, best_model['r2'], best_name

def main():
    # Erstelle Output-Verzeichnis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Daten einlesen und normieren
    print(f"Lese CSV-Datei: {CSV_FILE}")
    print(f"Normierung auf Probe: {REFERENCE_PROBE}")
    print(f"Wellenlänge: {WAVELENGTH}")
    print(f"Exponentielle Regression: {'Aktiviert' if USE_EXPONENTIAL else 'Deaktiviert'}")
    print(f"Power Law Regression: {'Aktiviert' if USE_POWER_LAW else 'Deaktiviert'}")
    print(f"Lineare Regression: {'Aktiviert' if USE_LINEAR else 'Deaktiviert'}")
    print("-" * 50)
    
    # Extrahiere CSV-Dateinamen ohne Endung
    csv_basename = os.path.splitext(os.path.basename(CSV_FILE))[0]
    
    df_original, df_normalized = read_and_normalize_data(
        CSV_FILE, REFERENCE_PROBE
    )

    print(f"df_normalized columns: {df_normalized.columns.tolist()}")
    
    # Erstelle Graphen für alle Kombinationen
    for messtyp, led, label in combinations:
        print(f"\nErstelle Graphen für {messtyp} - {led}...")
        
        fig1, fig2, r2, best_model_name = create_graphs(
            df_normalized, messtyp, led, WAVELENGTH, label
        )
        
        # Speichere Graphen
        filename1 = f'{csv_basename}_{label}_{WAVELENGTH}.png'
        filename2 = f'{csv_basename}_{label}_{WAVELENGTH}_reg.png'
        
        fig1.savefig(os.path.join(OUTPUT_DIR, filename1), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(OUTPUT_DIR, filename2), dpi=300, bbox_inches='tight')
        
        print(f"  Gespeichert: {filename1}")
        print(f"  Gespeichert: {filename2}")
        print(f"  Bestes Modell: {best_model_name}")
        print(f"  R² Wert: {r2:.4f}")
        
        # Gebe µa aus wenn verfügbar
        df_temp = df_normalized[(df_normalized['Messtyp'] == messtyp) & 
                                 (df_normalized['LED'] == led)].copy()
        df_temp['Concentration'] = df_temp['Text2'].apply(extract_concentration)
        df_temp = df_temp.sort_values('Concentration')
        X_temp = df_temp['Concentration'].values.reshape(-1, 1)
        y_temp = df_temp[WAVELENGTH].values
        models_temp, _, best_temp = fit_models(X_temp, y_temp)
        if 'mu_a' in best_temp:
            print(f"  µₐ = {best_temp['mu_a']:.4f} (µg/mL·mm)⁻¹")
        
        plt.close(fig1)
        plt.close(fig2)
    
    print("\n" + "=" * 50)
    print(f"Alle Graphen wurden gespeichert in: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
