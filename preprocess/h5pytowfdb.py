import os

import h5py
import numpy as np
import wfdb
import argparse

"""
python h5pytowfdb.py /path/to/folder/with/h5/files --fs 500
"""

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def find_dataset(group):
    """
    Automatically find the dataaset inside the h5 file.
    """

    datasets = []

    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            datasets.append(name)
    
    group.visititems(visitor)

    if not datasets:
        raise ValueError("No dataset found in H5 file.")
    
    return datasets[0]

def convert_h5_to_wfdb(h5_path, output_dir, fs):
    filename = os.path.splitext(os.path.basename(h5_path))[0]

    print(f"\nProcessing: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        dataset_name = find_dataset(f)

        print(f"Using dataset: {dataset_name}")
        
        signal = f[dataset_name][:]

    signal = np.asarray(signal, dtype=np.float64)

    # Ensure 2D
    if signal.ndim == 1:
        signal = signal.reshape(-1,1)

    # Ensure shape = (samples, channels)
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T
    
    num_channels = signal.shape[1]

    output_path = os.path.join(output_dir, filename)

    wfdb.wrsamp(
        record_name=filename,
        write_dir=output_dir,
        fs=fs,
        fmt=['16'] * num_channels,
        adc_gain=[1000] * num_channels,
        baseline=[0] * num_channels,
        units=['mV'] * num_channels,
        sig_name=STANDARD_12_LEADS if num_channels == 12 else [f"Lead{i+1}" for i in range(num_channels)],
        p_signal=signal
    )

    print(f"Create:")
    print(f" {output_path}.dat")
    print(f" {output_path}.hea")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all ecg .h5 files in a folder into WFDB .dat/.hea files"
    )

    parser.add_argument(
        "folder",
        help="Path to folder containing .h5 files"
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=500,
        help="Sampling frequency for the output WFDB files (default: 500)"
    )

    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    
    if not os.path.isdir(folder):
        print("Invalid folder path.")
        return
    
    h5_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".h5")
    ]

    if not h5_files:
        print("No .h5 files found.")
        return
    
    print(f"Found {len(h5_files)} .h5 files.")

    for h5_file in h5_files:
        try:
            convert_h5_to_wfdb(h5_file, folder, args.fs)
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

if __name__ == "__main__":
    main()
