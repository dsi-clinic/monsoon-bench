import os
import sys

import numpy as np

from examples.paper_figures.data_utils import save_all_data
from examples.paper_figures.fig3_utils import load_fig3_data


def _find_dirs_with_name(root, keyword, exclude=None):
    """Return all subdirectories under root whose name contains keyword (case-insensitive)."""
    matches = []
    keyword = keyword.lower()
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            name = d.lower()
            if keyword in name:
                if exclude is None or exclude.lower() not in name:
                    matches.append(os.path.join(dirpath, d))
    return matches


def _count_nc_files(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(".nc"))
    return count


def _prompt_user_dir(prompt):
    while True:
        path = input(prompt).strip()
        if os.path.isdir(path):
            return path
        print(f"  '{path}' is not a valid directory. Please try again.")


def _prompt_user_file(prompt):
    while True:
        path = input(prompt).strip()
        if os.path.isfile(path):
            return path
        print(f"  '{path}' is not a valid file. Please try again.")


def _confirm(found_label, found_value):
    answer = input(f"  Located {found_label}: {found_value}. Is this correct? (y/n): ").strip().lower()
    return answer == "y"


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------

def _resolve_imd_folder(data_dir):
    candidates = _find_dirs_with_name(data_dir, "imd")
    for candidate in candidates:
        # Check subdirectories for one with "4" in its name that has >20 .nc files
        subdirs = [
            os.path.join(candidate, d)
            for d in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, d))
        ]
        subdirs_with_4 = [s for s in subdirs if "4" in os.path.basename(s)]

        # Prefer subdirectory with "4"
        for sub in subdirs_with_4:
            if _count_nc_files(sub) > 20:
                return sub

        # Otherwise check all subdirs
        for sub in subdirs:
            if _count_nc_files(sub) > 20:
                return sub

        # Check the candidate directory itself
        if _count_nc_files(candidate) > 20:
            return candidate

    return None


def _resolve_thres_file(data_dir):
    candidates = _find_dirs_with_name(data_dir, "imd")
    for candidate in candidates:
        nc4_files = [
            os.path.join(candidate, f)
            for f in os.listdir(candidate)
            if f.endswith(".nc4")
        ]
        if len(nc4_files) == 1:
            return nc4_files[0]
    return None


def _resolve_shpfile_path(data_dir):
    for dirpath, _, files in os.walk(data_dir):
        if any(f.endswith(".shp") for f in files):
            return dirpath
    return None


def _resolve_model_path(data_dir, keyword, exclude=None, alt_keyword=None):
    candidates = _find_dirs_with_name(data_dir, keyword, exclude=exclude)
    if not candidates and alt_keyword:
        candidates = _find_dirs_with_name(data_dir, alt_keyword, exclude=exclude)
    if not candidates:
        return None
    # If multiple hits, pick the first one (closest to root)
    chosen = candidates[0]
    # Check for a subdirectory containing "4"
    subdirs = [
        os.path.join(chosen, d)
        for d in os.listdir(chosen)
        if os.path.isdir(os.path.join(chosen, d))
    ]
    subdirs_with_4 = [s for s in subdirs if "4" in os.path.basename(s)]
    if subdirs_with_4:
        chosen = subdirs_with_4[0]
    return chosen


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Ask for data directory
    data_dir = input("Enter the path to your data directory: ").strip()
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a valid directory.")
        sys.exit(1)

    cwd = os.getcwd()

    # ------------------------------------------------------------------
    # imd_folder
    # ------------------------------------------------------------------
    imd_folder = _resolve_imd_folder(data_dir)
    if imd_folder and _confirm("imd_folder", imd_folder):
        pass
    else:
        imd_folder = _prompt_user_dir("Enter the path to the imd folder: ")

    # ------------------------------------------------------------------
    # thres_file
    # ------------------------------------------------------------------
    thres_file = _resolve_thres_file(data_dir)
    if thres_file and _confirm("thres_file", thres_file):
        pass
    else:
        thres_file = _prompt_user_file("Enter the path to the threshold file: ")

    # ------------------------------------------------------------------
    # shpfile_path
    # ------------------------------------------------------------------
    shpfile_path = _resolve_shpfile_path(data_dir)
    if shpfile_path and _confirm("shpfile_path", shpfile_path):
        pass
    else:
        shpfile_path = _prompt_user_dir("Enter the path to the directory containing the shapefile: ")

    # ------------------------------------------------------------------
    # output_dir
    # ------------------------------------------------------------------
    output_dir = f"{cwd}/output"
    change = input(f"  Default output directory is '{output_dir}'. Do you want to change it? (y/n): ").strip().lower()
    if change == "y":
        output_dir = input("Enter the desired output directory: ").strip()
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "years": np.arange(2019, 2025),
        "extended_years": np.concatenate((np.arange(1965, 1979), np.arange(2019, 2025))),
        "common_years": np.arange(2004, 2022),
        "imd_folder": imd_folder,
        "thres_file": thres_file,
        "shpfile_path": shpfile_path,
        "output_dir": output_dir,
    }

    # ------------------------------------------------------------------
    # model_paths
    # ------------------------------------------------------------------
    model_search = {
        "IFS":      ("ifs",       "aifs", None),
        "AIFS":     ("aifs",      None,   None),
        "FuXi":     ("fuxi",      "s2s",  None),
        "Graphcast":("graphcast",  None,   None),
        "GenCast":  ("gencast",    None,   None),
        "FuXi-S2S": ("s2s",       None,   None),
        "NGCM":     ("ngcm",       None,   "neuralgcm"),
    }

    model_paths = {}
    for model_name, (keyword, exclude, alt_keyword) in model_search.items():
        found = _resolve_model_path(data_dir, keyword, exclude=exclude, alt_keyword=alt_keyword)
        if found and _confirm(f"directory for {model_name}", found):
            model_paths[model_name] = found
        else:
            model_paths[model_name] = _prompt_user_dir(f"Enter the directory for {model_name}: ")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    save_all_data(model_paths, config)
    load_fig3_data(model_paths, config)
