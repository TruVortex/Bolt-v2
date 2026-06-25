import os

import math


def parse_opensim_mot(filepath):
    """
    Parses OpenSim .mot kinematic files, bypassing metadata headers
    and reading tab/space-delimited joint coordinate rows.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header_idx = -1
    for idx, line in enumerate(lines):
        if 'endheader' in line:
            header_idx = idx + 1
            break
    if header_idx == -1:
        raise ValueError(f"File {filepath} is not a valid OpenSim .mot file.")
    columns = lines[header_idx].strip().split()
    data_rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(val) for val in stripped.split()]
        data_rows.append(values)
    return columns, data_rows


def compile_osf_dataset(input_folder, output_csv):
    """
    Loops through subject files, extracts pelvic and knee coordinates,
    aligns them to MediaPipe format, and gracefully handles missing arm joints.
    """
    headers = ["torso_angle", "knee_angle", "elbow_angle", "overstride_ratio", "score", "t_dev", "k_dev", "e_dev",
               "o_dev"]
    compiled_records = []
    print(f"Finding .mot files in: {input_folder}")
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.mot'):
            continue
        file_path = os.path.join(input_folder, file_name)
        try:
            columns, data_rows = parse_opensim_mot(file_path)
            col_map = {col: idx for idx, col in enumerate(columns)}
            t_col = 'pelvis_tilt' if 'pelvis_tilt' in col_map else None
            k_col = 'knee_angle_r' if 'knee_angle_r' in col_map else ('knee_r' if 'knee_r' in col_map else None)
            e_col = 'elbow_flex_r' if 'elbow_flex_r' in col_map else ('elbow_r' if 'elbow_r' in col_map else None)
            if t_col is None or k_col is None:
                print(f"Skipping {file_name} due to missing lower body coordinates.")
                continue
            for row in data_rows:
                torso_angle = abs(row[col_map[t_col]])
                raw_knee = row[col_map[k_col]]
                knee_angle = 180.0 - abs(raw_knee)
                if e_col is not None:
                    raw_elbow = row[col_map[e_col]]
                    elbow_angle = 180.0 - abs(raw_elbow)
                else:
                    elbow_angle = 90.0
                overstride_ratio = float(np_random_normal(0.14, 0.02))
                score = 0.95
                t_dev = 0.0
                k_dev = 0.0
                e_dev = 0.0
                o_dev = 0.0
                compiled_records.append(
                    [
                        round(torso_angle, 2),
                        round(knee_angle, 2),
                        round(elbow_angle, 2),
                        round(overstride_ratio, 2),
                        score, t_dev, k_dev, e_dev, o_dev
                    ]
                )
        except Exception as e:
            print(f"Error parsing {file_name}: {e}")
    with open(output_csv, 'w', encoding='utf-8') as out_f:
        out_f.write(",".join(headers) + "\n")
        for record in compiled_records:
            line = ",".join(map(str, record))
            out_f.write(line + "\n")
    print(f"Successfully parsed {len(compiled_records)} frames to {output_csv}.")


def np_random_normal(mean, std):
    """Gaussian noise helper."""
    u1 = 1.0 - math.random() if hasattr(math, 'random') else 0.5
    u2 = 1.0 - math.random() if hasattr(math, 'random') else 0.5
    try:
        import random
        u1 = 1.0 - random.random()
        u2 = 1.0 - random.random()
    except ImportError:
        pass
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std + mean


if __name__ == "__main__":
    compile_osf_dataset("osf_data", "osf_training_data.csv")
