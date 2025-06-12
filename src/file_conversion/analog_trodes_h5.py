import pandas as pd
import numpy as np
import re
import os
import warnings
from collections import defaultdict
import pathlib
import subprocess
import glob
import h5py
import spikeinterface.preprocessing as sp
import spikeinterface.extractors as se


def main(datapath):
    rec_files = find_merged_rec_files(datapath)
    print(f"Found {len(rec_files)} merged.rec files.")

    # Create output directory in project root
    output_dir = os.path.join(os.path.dirname(__file__), "h5_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for rec_file in rec_files:
        fname = os.path.splitext(os.path.basename(rec_file))[0]
        try:
            metadata = get_metadata_from_rec_name(fname)
        except ValueError as e:
            print(f"Skipping {rec_file}: {e}")
            continue

        ekg_signal, resp_signal = get_data_from_rec_file(rec_file)

        output_h5 = os.path.join(output_dir, fname + ".h5")
        export_to_h5(output_h5, ekg_signal, resp_signal, metadata)
        print(f"Exported: {output_h5}")



def get_metadata_from_rec_name(rec_name):
    pattern = r"sub[j]?_(\d+)_(\d+)_([a-zA-Z]+)(?:_d\d+)?_(\d{8})_(\d{6})_merged"
    match = re.match(pattern, rec_name)
    if not match:
        raise ValueError(f"Filename {rec_name} doesn't match expected pattern!")
    subject, session, condition, date, time = match.groups()
    return {
        "subject": int(subject),
        "session": int(session),
        "condition": condition,
        "date": date,
        "time": time
    }


def find_merged_rec_files(datapath):
    rec_files = []
    for root, dirs, files in os.walk(datapath):
        for f in files:
            if f.endswith('merged.rec'):
                rec_files.append(os.path.join(root, f))
    return rec_files


def get_data_from_rec_file(rec_file):
    try:
        ekg = se.read_spikegadgets(rec_file, stream_id='trodes')
        ekg_signal = ekg.get_traces(channel_ids=['21']).flatten()
    except Exception as e:
        ekg_signal = None
        print(f"Could not extract EKG: {e}")

    try:
        # Pass the folder containing the .rec file, not the rec file itself!
        parent_folder = os.path.dirname(rec_file)
        resp = organize_all_trodes_export(parent_folder)
        # The key is the folder name, not the rec file name
        fname = os.path.basename(parent_folder)
        resp_data = resp[fname]['analog']['analog_Controller_Ain1']['data']
        resp_signal = resp_data['voltage'].astype(float)
    except Exception as e:
        resp_signal = None
        print(f"Could not extract RESP: {e}")

    return ekg_signal, resp_signal



def export_to_h5(output_path, ekg_signal, resp_signal, metadata):
    with h5py.File(output_path, 'w') as f:
        if ekg_signal is not None:
            f.create_dataset('ekg', data=ekg_signal)
        if resp_signal is not None:
            f.create_dataset('resp', data=resp_signal)
        meta_grp = f.create_group('metadata')
        for k, v in metadata.items():
            meta_grp.attrs[k] = v


# ---------------------------------------------------------------------
def parse_fields(field_str):
    """
    Parses a string of fields into a numpy data type object.

    The input string should be formatted as '<fieldname num*type>' or
    '<fieldname type>'. This function parses the string, extracts the field
    names and data types, and creates a numpy data type object which can be
    used to read data from a binary file.

    Args:
        field_str (str): The string specifying the fields.

    Returns:
        np.dtype: A numpy data type object that describes the structure of
                  the data.

    Raises:
        SystemExit: If the provided field type is not a valid numpy data type.
    """

    # Clean up the string and split it into components
    components = re.split("\s", re.sub(r"\>\<|\>|\<", " ", field_str).strip())

    dtype_spec = []  # Will hold tuples to specify the numpy data type

    # Iterate over pairs of components (field name and type)
    for i in range(0, len(components), 2):
        field_name = components[i]

        # Default values
        repeat_count = 1
        field_type_str = "uint32"

        # If the field type string contains a '*', it indicates a repeat count
        if "*" in components[i + 1]:
            split_types = re.split("\*", components[i + 1])
            # Handle both 'num*type' and 'type*num'
            field_type_str = split_types[split_types[0].isdigit()]
            repeat_count = int(split_types[split_types[1].isdigit()])
        else:
            field_type_str = components[i + 1]

        # Convert the field type string to an actual numpy data type
        try:
            field_type = getattr(np, field_type_str)
        except AttributeError:
            print(f"{field_type_str} is not a valid field type.")
            exit(1)
        else:
            dtype_spec.append((str(field_name), field_type, repeat_count))

    return np.dtype(dtype_spec)


def read_trodes_extracted_data_file(filename):
    """
    Reads the content of a Trodes extracted data file.

    This function opens a Trodes file, reads the settings, parses them into a dictionary,
    and then reads the remaining data in the file as a numpy array according to the
    data types specified in the settings. If the settings block does not start correctly,
    it raises an Exception.

    Args:
        filename (str): The path to the Trodes file to be read.

    Returns:
        dict: A dictionary where keys are settings field names and values are the
              corresponding setting values. The actual data from the file is stored
              under the 'data' key as a numpy array.

    Raises:
        Exception: If the settings block in the file does not start with '<Start settings>'.
    """
    with open(filename, "rb") as f:
        # The first line of the file should start the settings block
        if f.readline().decode("ascii").strip() != "<Start settings>":
            raise Exception("Settings format not supported")

        # Flag indicating we're reading the settings block
        fields = True
        # Dictionary to hold the settings fields and values
        fields_text = {}

        # Iterate over the lines in the file
        for line in f:
            # If we're still reading the settings block
            if fields:
                line = line.decode("ascii").strip()
                # If we've not reached the end of the settings block, continue reading fields
                if line != "<End settings>":
                    key, value = line.split(": ")
                    fields_text.update({key.lower(): value})
                # If we've reached the end of the settings block, stop reading fields
                else:
                    fields = False
                    # Parse the 'fields' setting to get the data type
                    dt = parse_fields(fields_text["fields"])
                    fields_text["data"] = np.zeros([1], dtype=dt)
                    break

        # Read the remaining data from the file using the parsed data type
        dt = parse_fields(fields_text["fields"])
        data = np.fromfile(f, dt)
        fields_text.update({"data": data})
        fields_text.update({"filename": os.path.basename(filename)})
        return fields_text


def organize_single_trodes_export(dir_path, skip_raw_group0=True):
    """
    Organizes Trodes data files in a given directory. The data is stored in a dictionary.
    The key is the penultimate (second to last) part of the file name (i.e., the part before the last dot in the file name).
    The values in the dictionary are the parsed data from the Trodes files.

    Args:
        dir_path (str): The path to the directory containing the Trodes files.
        skip_raw_group0(bool): To skip the "raw_group0" file which contains the raw signal which uses a lot of memory
    Returns:
        dict: A dictionary with organized Trodes file data.
    """
    # Initialize dictionary to store results
    result = {}

    # Iterate over all files in the directory
    for file_name in os.listdir(dir_path):

        if skip_raw_group0 and "raw_group0" in file_name:
            continue
        # Attempt to parse each file and store the data in the dictionary
        try:
            # Extract second to last part of the file name
            sub_dir_name = file_name.rsplit(".", 2)[-2]
            # Parse Trodes file and store the data
            result[sub_dir_name] = read_trodes_extracted_data_file(os.path.join(dir_path, file_name))

        # Skip files that cause errors during parsing
        except Exception as e:
            print(f"Skipping file {file_name} due to error: {e}")
            continue

    return result

def organize_all_trodes_export(dir_path):
    """
    Organize Trodes files in subdirectories, mapping .analog/.time/etc. folders to canonical keys.
    """
    result = defaultdict(dict)

    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            # Map common suffixes to canonical keys
            if ".analog" in sub_dir_name:
                canonical_key = 'analog'
            elif ".time" in sub_dir_name:
                canonical_key = 'time'
            elif ".timestampoffset" in sub_dir_name:
                canonical_key = 'timestampoffset'
            else:
                # fallback: use last part after dot
                canonical_key = sub_dir_name.split('.')[-1]
            try:
                result[os.path.basename(dir_path)][canonical_key] = organize_single_trodes_export(sub_dir_path)
            except Exception as e:
                print(f"Error processing subdirectory {sub_dir_path}: {e}")
                continue
    return result

#-----------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print("Starting conversion...")

    # Change this to your actual data path
    datapath = r"C:\Users\thoma\Code\ResearchCode\spike_to_h5\ekg_resp_data"
    
    main(datapath)
    print("Done.")
