import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.container import ErrorbarContainer

# ===== CONFIG =====
CSV_FILEPATH = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Hefe\CSV\Messreihe Hefe Lena Messvorschrift_spectrum_merged.csv"
OUTPUT_DIR = r"g:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 10mm\Datenauswertung\Hefe\Spektrum2"
REFERENCE_PROBE = "I"
PROBENART = "Hefe"
# ==================

# Spectral channels to normalize
SPECTRAL_CHANNELS = ['405nm', '425nm', '450nm', '475nm', '515nm', '550nm', 
                     '555nm', '600nm', '640nm', '690nm', '745nm', '855nm']

def read_and_prepare_data(filepath):
    """Read CSV and prepare dataframe."""
    df = pd.read_csv(filepath)
    return df

def normalize_to_reference(group_df, reference_probe):
    """Normalize spectral channels to reference probe within a group."""
    # Find reference row(s) in this group
    ref_rows = group_df[group_df['Text1'] == reference_probe]
    
    if len(ref_rows) == 0:
        raise ValueError(f"Reference probe {reference_probe} not found in group")
    
    # Calculate mean reference values for each spectral channel
    ref_values = ref_rows[SPECTRAL_CHANNELS].mean()
    
    # Normalize all rows in group
    normalized_df = group_df.copy()
    for channel in SPECTRAL_CHANNELS:
        normalized_df[channel] = group_df[channel] / ref_values[channel]
    
    return normalized_df

def process_measurement_type(df, messtyp):
    """Process data for one measurement type."""
    # Filter by measurement type
    df_messtyp = df[df['Messtyp'] == messtyp].copy()
    
    # Create a group identifier based on the sequence of Text1
    # Each time we see the reference probe, increment the group counter
    group_id = 0
    group_ids = []
    
    for i, text1 in enumerate(df_messtyp['Text1'].values):
        group_ids.append(group_id)
        if text1 == REFERENCE_PROBE:
            group_id += 1
    
    df_messtyp['group_id'] = group_ids
    
    # Normalize each group separately
    normalized_groups = []
    for gid in df_messtyp['group_id'].unique():
        group = df_messtyp[df_messtyp['group_id'] == gid]
        normalized_group = normalize_to_reference(group, REFERENCE_PROBE)
        normalized_groups.append(normalized_group)
    
    df_normalized = pd.concat(normalized_groups, ignore_index=True)
    
    # Calculate statistics for each Text1 across all groups
    stats = []
    print(f"\n--- Spannweiten bei 600nm ({messtyp}) ---")
    for text1 in df_normalized['Text1'].unique():
        text1_data = df_normalized[df_normalized['Text1'] == text1]
        
        means = text1_data[SPECTRAL_CHANNELS].mean()
        # Calculate range (max - min) for each channel
        mins = text1_data[SPECTRAL_CHANNELS].min()
        maxs = text1_data[SPECTRAL_CHANNELS].max()
        ranges = maxs - mins
        
        # Print range for 600nm specifically
        if '600nm' in ranges:
            print(f"Probe {text1}: {ranges['600nm']:.6f}")

        stats.append({
            'Text1': text1,
            'Text2': text1_data['Text2'].iloc[0],  # Get concentration label
            'means': means,
            'ranges': ranges
        })
    
    return stats

def plot_spectrum(stats, messtyp, output_dir, base_filename):
    """Create plot with error bars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use channel names as x-axis (categorical with equal spacing)
    x_positions = np.arange(len(SPECTRAL_CHANNELS))
    
    # Sort stats by concentration (reverse order to have highest first)
    stats_sorted = sorted(stats, key=lambda x: x['Text1'])
    # Plot each probe
    for stat in stats_sorted:
        means = stat['means'].values
        ranges = stat['ranges'].values
        label = f"{stat['Text1']}: {stat['Text2']}"
        
        # Use half of range as error bar (symmetric around mean)
        yerr = ranges / 2
        
        # Plot with error bars but hide them in legend
        line = ax.errorbar(x_positions, means, yerr=yerr, marker='o', 
                   capsize=2, capthick=0.7, label=label, linewidth=2, markersize=6,
                   elinewidth=0.7, alpha=1.0, errorevery=1)
    
    # Set x-axis to show only numeric wavelength values
    wavelength_values = [ch.replace('nm', '') for ch in SPECTRAL_CHANNELS]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(wavelength_values)
    
    ax.set_xlabel('Wellenlänge (nm)', fontsize=12)
    ax.set_ylabel(f'Normalisierte Intensität (auf Probe {REFERENCE_PROBE}) (AU)', fontsize=12)
    ax.set_title(f'{PROBENART} - {messtyp} - LED Intensität [100%, 5%, 0.1%]', fontsize=14, fontweight='bold')
    
    # Create legend with custom handler to remove error bars from data entries
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for i, handle in enumerate(handles):
        # Extract color from ErrorbarContainer
        color = handle.lines[0].get_color()
        # Create new handle with only line and marker, no error bars
        new_handle = plt.Line2D([], [], color=color, 
                               marker='o', linewidth=2, markersize=6)
        new_handles.append(new_handle)
    
    # Create dummy errorbar container for legend only (not plotted)
    # We create it on a temporary invisible axis
    temp_fig, temp_ax = plt.subplots()
    dummy_eb = temp_ax.errorbar([0], [0], yerr=[0.5], marker='o', color='black',
                                capsize=4, capthick=1, linewidth=0, markersize=6,
                                elinewidth=1)
    plt.close(temp_fig)
    
    
    new_handles.append(dummy_eb)
    labels.append('Spannweite min-max')
    
    # Use handler_map to keep full errorbar in legend for the dummy entry
    ax.legend(new_handles, labels, loc='lower right', 
             handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0, yerr_size=1)})
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save files with custom naming
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine suffix based on measurement type
    suffix = "_ref" if messtyp == "Reflexion" else "_trans"
    output_name = f"{base_filename}{suffix}"
    
    plt.savefig(output_path / f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{output_name}.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Main processing function."""
    print("Reading CSV file...")
    df = read_and_prepare_data(CSV_FILEPATH)
    
    # Extract base filename from CSV path (up to "Messvorschrift")
    csv_filename = Path(CSV_FILEPATH).stem  # Get filename without extension
    if "Messvorschrift" in csv_filename:
        base_filename = csv_filename.split("Messvorschrift")[0].rstrip(" _")
    else:
        base_filename = csv_filename
    
    # Process each measurement type
    for messtyp in ['Reflexion', 'Transmission']:
        print(f"\nProcessing {messtyp}...")
        stats = process_measurement_type(df, messtyp)
        
        print(f"Creating plot for {messtyp}...")
        plot_spectrum(stats, messtyp, OUTPUT_DIR, base_filename)
        
        print(f"Saved plots to {OUTPUT_DIR}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
