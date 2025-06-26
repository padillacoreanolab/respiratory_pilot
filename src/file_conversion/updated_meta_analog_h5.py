import os
import re
import h5py
import numpy as np
from collections import defaultdict
import spikeinterface.extractors as se


def find_merged_rec_files(datapath):
    rec_files = []
    for root, dirs, files in os.walk(datapath):
        for f in files:
            if f.endswith('merged.rec'):
                rec_files.append(os.path.join(root, f))
    return rec_files


def get_metadata_from_rec_name(rec_name):
    """
    Extracts structured metadata from a filename like:
    BLRI_s1_1_p5_2_nRB3_20250622_141846_merged

    Returns dict with subject ID, agent info, date/time, etc.
    """
    pattern = r"(BLRI|RI\d)_s(\d+)_(\d+)_p(\d+)_(\d+)_(n?RB\d+)_(\d{8})_(\d{6})_merged"
    match = re.match(pattern, rec_name)
    if not match:
        raise ValueError(f"Filename {rec_name} doesn't match expected pattern!")

    trial_type, subj_major, subj_minor, pos_major, pos_minor, neg_agent, date, time = match.groups()

    return {
        "trial_type": trial_type,  # BLRI or RI1
        "subject_id": f"{subj_major}.{subj_minor}",  # e.g., 1.1
        "positive_agent_id": f"{pos_major}.{pos_minor}",  # e.g., 5.2
        "negative_agent_id": neg_agent.replace("n", ""),  # Strip 'n' if present
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",  # YYYY-MM-DD
        "time": f"{time[:2]}:{time[2:4]}:{time[4:]}"  # HH:MM:SS
    }



def read_trodes_extracted_data_file(filename):
    with open(filename, "rb") as f:
        if f.readline().decode("ascii").strip() != "<Start settings>":
            raise Exception("Settings format not supported")

        fields = True
        fields_text = {}
        for line in f:
            if fields:
                line = line.decode("ascii").strip()
                if line != "<End settings>":
                    key, value = line.split(": ")
                    fields_text[key.lower()] = value
                else:
                    fields = False
                    break

        dt = parse_fields(fields_text["fields"])
        data = np.fromfile(f, dt)
        fields_text["data"] = data
        return fields_text


def parse_fields(field_str):
    import re
    components = re.split("\s", re.sub(r"\>\<|\>|\<", " ", field_str).strip())
    dtype_spec = []

    for i in range(0, len(components), 2):
        field_name = components[i]
        repeat_count = 1
        field_type_str = components[i + 1]

        if "*" in field_type_str:
            parts = field_type_str.split("*")
            if parts[0].isdigit():
                repeat_count = int(parts[0])
                field_type_str = parts[1]
            else:
                repeat_count = int(parts[1])
                field_type_str = parts[0]

        dtype_spec.append((field_name, getattr(np, field_type_str), repeat_count))

    return np.dtype(dtype_spec)


def organize_single_trodes_export(dir_path):
    result = {}
    for file_name in os.listdir(dir_path):
        if "raw_group0" in file_name or file_name.endswith(".txt"):
            continue
        sub_key = file_name.rsplit(".", 2)[-2]
        try:
            result[sub_key] = read_trodes_extracted_data_file(os.path.join(dir_path, file_name))
        except Exception as e:
            print(f"Skipping {file_name}: {e}")
    return result


def organize_all_trodes_export(dir_path):
    result = defaultdict(dict)
    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            if ".analog" in sub_dir_name:
                key = "analog"
            elif ".time" in sub_dir_name:
                key = "time"
            else:
                key = sub_dir_name.split(".")[-1]
            try:
                result[os.path.basename(dir_path)][key] = organize_single_trodes_export(sub_dir_path)
            except Exception as e:
                print(f"Error reading {sub_dir_path}: {e}")
    return result


def get_data_from_rec_file(rec_file):
    ekg_signal, resp_signal = None, None
    ekg_meta, resp_meta = {}, {}

    # EKG from SpikeGadgets .rec via SpikeInterface
    try:
        ekg = se.read_spikegadgets(rec_file, stream_id='trodes')
        ekg_signal = ekg.get_traces(channel_ids=['21']).flatten()

        ekg_meta = {
            'sampling_frequency': ekg.get_sampling_frequency(),
            'channel_id': '21',
            'num_samples': len(ekg_signal),
            'duration_sec': len(ekg_signal) / ekg.get_sampling_frequency(),
            'stream_id': 'trodes'
        }
    except Exception as e:
        print(f"Could not extract EKG: {e}")

    # RESP from .analog folder
    try:
        parent_folder = os.path.dirname(rec_file)
        resp = organize_all_trodes_export(parent_folder)
        fname = os.path.basename(parent_folder)
        resp_group = resp[fname]['analog']['analog_ECU_Ain1']

        resp_signal = resp_group['data']['voltage'].astype(float)
        resp_sr = resp_group['sample_rate'][()]
        resp_meta = {
            'sampling_frequency': resp_sr,
            'start_time': resp_group.get('start_time', np.nan),
            'num_samples': len(resp_signal),
            'duration_sec': len(resp_signal) / resp_sr,
            'channel': 'analog_ECU_Ain1',
            'byte_order': resp_group.get('byte_order', 'unknown'),
            'units': 'volts'
        }
    except Exception as e:
        print(f"Could not extract RESP: {e}")

    return ekg_signal, ekg_meta, resp_signal, resp_meta



def export_to_h5(output_path, ekg_signal, resp_signal, metadata, ekg_meta, resp_meta):
    with h5py.File(output_path, 'w') as f:
        if ekg_signal is not None:
            f.create_dataset('ekg', data=ekg_signal)
        if resp_signal is not None:
            f.create_dataset('resp', data=resp_signal)

        # General metadata
        meta_grp = f.create_group('metadata')
        for k, v in metadata.items():
            meta_grp.attrs[k] = v

        # EKG-specific metadata
        ekg_grp = f.create_group('ekg_metadata')
        for k, v in ekg_meta.items():
            ekg_grp.attrs[k] = v

        # RESP-specific metadata
        resp_grp = f.create_group('resp_metadata')
        for k, v in resp_meta.items():
            resp_grp.attrs[k] = v



def convert_all_rec_to_h5(datapath, output_dir="h5_outputs"):
    rec_files = find_merged_rec_files(datapath)
    print(f"Found {len(rec_files)} merged.rec files.")

    os.makedirs(output_dir, exist_ok=True)


    for rec_file in rec_files:
        fname = os.path.splitext(os.path.basename(rec_file))[0]
        try:
            metadata = get_metadata_from_rec_name(fname)
        except ValueError as e:
            print(f"Skipping {rec_file}: {e}")
            continue

        ekg_signal, ekg_meta, resp_signal, resp_meta = get_data_from_rec_file(rec_file)

        output_h5 = os.path.join(output_dir, fname + ".h5")
        export_to_h5(output_h5, ekg_signal, resp_signal, metadata, ekg_meta, resp_meta)
        print(f"Exported: {output_h5}")


# Run this if executing the script
if __name__ == "__main__":
    data_root = r"C:\Users\thoma\UFL Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new"
    output_dir = r"C:\Users\thoma\Code\ResearchCode\respiratory_pilot\src\file_conversion\updated_meta_h5"
    convert_all_rec_to_h5(data_root, output_dir=output_dir)
