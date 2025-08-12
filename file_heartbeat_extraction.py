import pandas as pd
import sys
import os

def filter_heartbeat_and_normal(input_filename, output_filename=None):
    """
    Filters rows with labels 'cyberattack_ocpp16_dos_flooding_heartbeat' and 'normal'
    from a CSV file and saves the result to a new CSV file.

    Parameters:
        input_filename (str): Path to the input CSV file.
        output_filename (str, optional): Path to save the filtered output file.
                                         If not provided, '_filtered.csv' is appended to input filename.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: file '{input_filename}' not found.")
        return

    if 'label' not in df.columns:
        print("Error: the input file must contain a 'label' column.")
        return

    # Filter the relevant rows
    filtered_df = df[df['label'].isin([
        'cyberattack_ocpp16_dos_flooding_heartbeat',
        'normal'
    ])]

    # Determine the output filename
    if output_filename is None:
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_filtered{ext}"

    # Save the filtered data
    filtered_df.to_csv(output_filename, index=False)
    print(f"Filtered data saved to: {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_heartbeat.py <input_file.csv> [output_file.csv]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        filter_heartbeat_and_normal(input_file, output_file)
