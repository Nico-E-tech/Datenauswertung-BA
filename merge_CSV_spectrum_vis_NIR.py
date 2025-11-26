import pandas as pd
import os

# File paths
base_path = r'G:\Meine Ablage\Studium\6. Semester\BA Arbeit\M2 Messergebnisse 1mm\Datenauswertung\Hefe\CSV'
base_name = 'Messreihe Hefe Lena Messvorschrift'
output_dir = base_path

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List to store all merged dataframes
all_merged_dfs = []

# Process files A, B, C
for suffix in ['A', 'B', 'C']:
    file_path = os.path.join(base_path, f'{base_name} {suffix}.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warnung: Datei nicht gefunden: {file_path}")
        continue
    
    print(f"\nVerarbeite: {base_name} {suffix}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Create a copy for modification
    df_merged = df.copy()
    
    # Group by Text1 and Messtyp
    grouped = df.groupby(['Text1', 'Messtyp'])
    
    # Process each group
    for (text1, messtyp), group in grouped:
        # Find WHITE and NIR rows
        white_idx = group[group['LED'] == 'WHITE'].index
        nir_idx = group[group['LED'] == 'NIR'].index
        
        if len(white_idx) > 0 and len(nir_idx) > 0:
            # Get 855nm value from NIR row
            nir_value = df_merged.loc[nir_idx[0], '855nm']
            
            # Replace 855nm value in WHITE row
            df_merged.loc[white_idx[0], '855nm'] = nir_value
    
    # Keep only WHITE LED rows
    df_result = df_merged[df_merged['LED'] == 'WHITE'].copy()
    
    # Add to list
    all_merged_dfs.append(df_result)
    
    print(f"  Zeilen verarbeitet: {len(df_result)}")

# Combine all merged dataframes
if all_merged_dfs:
    combined_df = pd.concat(all_merged_dfs, ignore_index=True)
    
    # Save combined result
    output_filename = f"{base_name}_spectrum_merged.csv"
    output_path = os.path.join(output_dir, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nAlle Spektren zusammengeführt und gespeichert: {output_path}")
    print(f"Gesamt Zeilen: {len(combined_df)}")
    print(f"Dateien verarbeitet: {len(all_merged_dfs)}")
else:
    print("\nKeine Dateien gefunden oder verarbeitet.")
