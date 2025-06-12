import pandas as pd
import numpy as np
import re
import os
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
        resp = se.read_spikegadgets(rec_file, stream_id='ECU')
        resp_signal = resp.get_traces(channel_ids=['21']).flatten()
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
# change datapath to your folder with the .rec files
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting conversion...")
    datapath = r"C:\Users\thoma\Code\ResearchCode\spike_to_h5\ekg_resp_data"
    main(datapath)
    print("Done.")
