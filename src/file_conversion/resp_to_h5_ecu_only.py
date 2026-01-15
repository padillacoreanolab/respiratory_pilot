import os
import h5py
import numpy as np

# ============================================================
# Trodes binary reader (wiki-compatible, NumPy-safe)
# ============================================================

def parse_fields(field_str):
    import re

    components = re.split(r"\s+", re.sub(r"[<>]", " ", field_str).strip())
    dtype_spec = []

    i = 0
    while i < len(components):
        name = components[i]
        type_str = components[i + 1]
        count = 1

        if "*" in type_str:
            n, t = type_str.split("*")
            if n.isdigit():
                count = int(n)
                type_str = t
            else:
                count = int(t)
                type_str = n

        np_type = getattr(np, type_str)

        if count == 1:
            dtype_spec.append((name, np_type))
        else:
            dtype_spec.append((name, np_type, (count,)))

        i += 2

    return np.dtype(dtype_spec)


def read_trodes_dat_file(filepath):
    with open(filepath, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header != "<Start settings>":
            raise ValueError(f"{filepath} is not a Trodes-exported .dat file")

        settings = {}
        for line in f:
            line = line.decode("ascii").strip()
            if line == "<End settings>":
                break
            key, val = line.split(": ", 1)
            settings[key.lower()] = val

        dtype = parse_fields(settings["fields"])
        data = np.fromfile(f, dtype)

    return settings, data


# ============================================================
# Load ECU exports from ONE session folder
# ============================================================

def load_trodes_exports(session_dir):
    analog = {}
    dio = {}
    time = {}

    for item in os.listdir(session_dir):
        full = os.path.join(session_dir, item)
        if not os.path.isdir(full):
            continue

        lname = item.lower()

        # ---------- ANALOG ----------
        if lname.endswith(".analog"):
            for f in os.listdir(full):
                if f.lower().endswith(".dat"):
                    settings, data = read_trodes_dat_file(os.path.join(full, f))
                    analog[f.replace(".dat", "")] = {
                        "data": data,
                        "settings": settings
                    }

        # ---------- DIO ----------
        if lname.endswith(".dio"):
            for f in os.listdir(full):
                if f.lower().endswith(".dat"):
                    settings, data = read_trodes_dat_file(os.path.join(full, f))
                    dio[f.replace(".dat", "")] = {
                        "data": data,
                        "settings": settings
                    }

        # ---------- TIME ----------
        if lname.endswith(".time"):
            for f in os.listdir(full):
                if f.lower().endswith(".dat"):
                    settings, data = read_trodes_dat_file(os.path.join(full, f))
                    time[f.replace(".dat", "")] = {
                        "data": data,
                        "settings": settings
                    }

    return analog, dio, time


# ============================================================
# Write HDF5
# ============================================================

def write_h5(output_path, analog, dio, time):
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(output_path, "w") as f:

        grp = f.create_group("analog")
        for name, obj in analog.items():
            dset = grp.create_dataset(name, data=obj["data"])
            for k, v in obj["settings"].items():
                dset.attrs[k] = v

        grp = f.create_group("dio")
        for name, obj in dio.items():
            dset = grp.create_dataset(name, data=obj["data"])
            for k, v in obj["settings"].items():
                dset.attrs[k] = v

        grp = f.create_group("time")
        for name, obj in time.items():
            dset = grp.create_dataset(name, data=obj["data"])
            for k, v in obj["settings"].items():
                dset.attrs[k] = v


# ============================================================
# Batch convert ALL session folders in a day
# ============================================================

def convert_sessions(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for session in os.listdir(root_dir):
        session_dir = os.path.join(root_dir, session)
        if not os.path.isdir(session_dir):
            continue

        print(f"\nProcessing: {session}")

        analog, dio, time = load_trodes_exports(session_dir)

        if not (analog or dio or time):
            print("  No Trodes ECU exports found — skipping")
            continue

        session_name = session.replace(".rec", "")
        out_h5 = os.path.join(output_dir, f"{session_name}.h5")

        write_h5(out_h5, analog, dio, time)

        print(f"  Wrote {out_h5}")


# ============================================================
# Optional alias (for backward compatibility)
# ============================================================

def convert_all_rec_to_h5(root_dir, output_dir):
    convert_sessions(root_dir, output_dir)


# ============================================================
# Run from terminal (optional)
# ============================================================

if __name__ == "__main__":
    data_root = r"PATH_TO_PARENT_FOLDER_WITH_SESSION_FOLDERS"
    output_dir = r"PATH_TO_OUTPUT_H5_FOLDER"

    convert_sessions(data_root, output_dir)
