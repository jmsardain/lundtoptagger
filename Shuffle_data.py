import argparse
import os
import glob
import time
from datetime import timedelta
import gc
from operator import itemgetter

from typing import List, Optional
import pandas as pd

import uproot
import awkward as ak
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight, create_train_dataset_fulld_new_Ntrk_pt_weight_file
from tools.utils_config import recursive_update, parse_dot_args


print("Libraries loaded!")


def read_flat_root_arrays(root_files: List[str],
                          tree: str,
                          branches: List[str]) -> pd.DataFrame:
    """
    Read the root files and return a pandas DataFrame
    """
    frames = []
    for f in root_files:
        print(f"Reading {f}...")
        with uproot.open(f) as rf:
            arrs = rf[tree].arrays(branches, library='np')
            df = pd.DataFrame({b: arrs[b] for b in branches})
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

def ensure_label_values(df: pd.DataFrame, label_branch: str, labels: List[int], label_map: Optional[dict]):
    """
    Ensure that the label branch contains only the specified labels.
    If label_map is provided, map the labels before filtering.
    """
    if label_map is not None:
        df[label_branch] = df[label_branch].map(label_map)
    df = df[df[label_branch].isin(labels)]
    df = df.reset_index(drop=True)
    return df


# ---------------------- Weighting ----------------------
def compute_1d_flat_weights(values: np.ndarray, range_min: float, range_max: float, n_bins: int):
    """Compute weights for 1D flattening in given range."""
    bins = np.linspace(range_min, range_max, n_bins + 1)
    counts, edges = np.histogram(values, bins=bins)
    counts_safe = counts.astype(float)
    counts_safe[counts_safe == 0] = 1.0
    weights = 1.0 / counts_safe
    # normalize such that the 1d spectrum becomes flat, match to the maximum count
    weights *= np.max(counts)
    return edges, weights

def compute_2d_flat_weights(x: np.ndarray, y: np.ndarray,
                            xrange: tuple, nx: int,
                            yrange: tuple, ny: int):
    """Compute weights for 2D flattening in given ranges."""
    xbins = np.linspace(xrange[0], xrange[1], nx + 1)
    ybins = np.linspace(yrange[0], yrange[1], ny + 1)
    counts, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    counts_safe = counts.astype(float)
    counts_safe[counts_safe == 0] = 1.0
    weights2d = 1.0 / counts_safe
    weights2d *= np.max(counts)
    return xedges, yedges, weights2d

def assign_weights(values, edges, weights):
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, len(weights)-1)
    return weights[idx]

def assign_2d_weights(x, y, xedges, yedges, weights2d):
    xi = np.digitize(x, xedges) - 1
    yi = np.digitize(y, yedges) - 1
    xi = np.clip(xi, 0, weights2d.shape[0]-1)
    yi = np.clip(yi, 0, weights2d.shape[1]-1)
    return weights2d[xi, yi]

def save_split_histograms(df, name, config, weighted=True):
    """
    df: pandas DataFrame for this split
    weighted: if True, use weights; else uniform weights
    """
    out_dir = os.path.join(config["output_dir"], name)
    os.makedirs(out_dir, exist_ok=True)

    weight_col = "weights" if weighted else None
    suffix = "weighted" if weighted else "raw"

    pt_edges = np.linspace(config["pt_range"][0], config["pt_range"][1], config["pt_binN"]+1)
    mass_edges = np.linspace(config["mass_range"][0], config["mass_range"][1], config["mass_binN"]+1)

    for label in config["labels"]:
        sel_label = df[df[config["branch_label"]] == label]

        # 1D pt histogram
        plt.figure()
        plt.hist(sel_label[config["branch_pt"]], bins=pt_edges, weights=sel_label[weight_col] if weighted else None,
                 histtype='step', lw=2)
        plt.title(f"Class {label} pt ({suffix})")
        plt.xlabel("pt [GeV]")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{name}_class{label}_pt_{suffix}.png"))
        plt.close()

        # 1D mass histogram
        plt.figure()
        plt.hist(sel_label[config["branch_mass"]], bins=mass_edges, weights=sel_label[weight_col] if weighted else None,
                 histtype='step', lw=2)
        plt.title(f"Class {label} mass ({suffix})")
        plt.xlabel("mass [GeV]")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{name}_class{label}_mass_{suffix}.png"))
        plt.close()

        # 2D pt-mass histogram
        plt.figure()
        plt.hist2d(sel_label[config["branch_pt"]],
                   sel_label[config["branch_mass"]],
                   bins=[pt_edges, mass_edges],
                   weights=sel_label[weight_col] if weighted else None,
                   cmap='viridis')
        plt.colorbar(label='Counts')
        plt.xlabel("pt [GeV]")
        plt.ylabel("mass [GeV]")
        plt.title(f"Class {label} pt vs mass ({suffix})")
        plt.savefig(os.path.join(out_dir, f"{name}_class{label}_pt_mass_{suffix}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Reweight, shuffle, balance, and split data into train/test")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    parser.add_argument('--override', nargs='*', default=[], help='Overrides in the form key.subkey=value')
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)


    filepath_placeholder_vals = dict( # Dummy values for formatting
        # sample = config['data']['sample'],
        # kT_cut = kT_selection
    )

    paths_to_file_root = config['data']['paths_to_file_root']
    if isinstance(paths_to_file_root, str):
        # paths_to_file_root can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        paths_to_file_root = [paths_to_file_root]
    files_root = []
    for file_path in paths_to_file_root:
        file_path = file_path.format(**filepath_placeholder_vals)
        files_root.extend(glob.glob(file_path))
    files_root.sort()
    print ("paths_to_file_root:", paths_to_file_root)
    print ("root files:", files_root)

    paths_to_file_graphs = config['data']['paths_to_file_graphs']
    if isinstance(paths_to_file_graphs, str):
        # paths_to_file_graphs can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        paths_to_file_graphs = [paths_to_file_graphs]
    files_graphs = []
    for file_path in paths_to_file_graphs:
        file_path = file_path.format(**filepath_placeholder_vals)
        files_graphs.extend(glob.glob(file_path))
    files_graphs.sort()
    print ("paths_to_file_graphs:", paths_to_file_graphs)
    print ("graph files:", files_graphs)

    if len(files_graphs) != len(files_root):
        raise RuntimeError(f"Graph files ({len(files_graphs)}) != ROOT files ({len(files_root)})")
    print(f"Matched {len(files_graphs)} ROOT ↔ graph file pairs.")

    # Small files for histogramming
    small_df = read_flat_root_arrays(files_root, config["tree_name"],
                                     [config["branch_pt"], config["branch_mass"], config["branch_label"]])
    small_df = ensure_label_values(small_df, config["branch_label"], config["labels"], config["label_map"])

    hist = {}
    for label in config["labels"]:
        df_label = small_df[small_df[config["branch_label"]] == label]
        print(f"Class {label}: {len(df_label)} jets for histogramming")
        if config["reweight_mode"] == "flat_pt":
            print("Computing pt flattening weights...")
            edges, w = compute_1d_flat_weights(
                df_label[config["branch_pt"]],
                config["pt_range"][0], config["pt_range"][1], config["pt_binN"]
            )
            hist[label] = {"pt_edges": edges, "pt_weights": w}

        elif config["reweight_mode"] == "flat_mass":
            print("Computing mass flattening weights...")
            edges, w = compute_1d_flat_weights(
                df_label[config["branch_mass"]],
                config["mass_range"][0], config["mass_range"][1], config["mass_binN"]
            )
            hist[label] = {"mass_edges": edges, "mass_weights": w}

        elif config["reweight_mode"] == "flat_pt_mass_2d":
            print("Computing 2D pt-mass flattening weights...")
            xedges, yedges, w2d = compute_2d_flat_weights(
                df_label[config["branch_pt"]],
                df_label[config["branch_mass"]],
                config["pt_range"], config["pt_binN"],
                config["mass_range"], config["mass_binN"]
            )
            hist[label] = {"pt_edges": xedges, "mass_edges": yedges, "pt_mass_weights2d": w2d}

    # np.savez_compressed(os.path.join(config["output_dir"], config["histogram_filename"]), **hist)

    # Full dataset with file_id/local_index, early stopper
    full_dfs = []
    class_counts = {label: 0 for label in config["labels"]}
    total_needed = int(float(config["n_jets_per_class"]))

    for fid, root_file in enumerate(files_root):
        with uproot.open(root_file) as rf:
            arrs = rf[config["tree_name"]].arrays(
                [config["branch_pt"], config["branch_mass"], config["branch_label"]],
                library='np'
            )
            df = pd.DataFrame(arrs)
            df["_file_id"] = fid
            df["_local_index"] = np.arange(len(df))

            # Only keep jets that still need to reach n_jets_per_class
            keep_mask = df[config["branch_label"]].apply(
                lambda lbl: class_counts[lbl] < total_needed
            )
            df = df[keep_mask]
            if df.empty:
                continue

            # Update class counters
            for lbl, count in df[config["branch_label"]].value_counts().items():
                class_counts[lbl] += count

            full_dfs.append(df)

            # Check if all classes have enough jets
            if all(cnt >= total_needed for cnt in class_counts.values()):
                print("Reached required jets for all classes. Stopping early.")
                break

    full_df = pd.concat(full_dfs, ignore_index=True)
    full_df = ensure_label_values(full_df, config["branch_label"], config["labels"], config["label_map"])

    # Sample equal per class
    n_target = int(float(config["n_jets_per_class"]))
    selected = []
    for label in config["labels"]:
        df_lbl = full_df[full_df[config["branch_label"]] == label]
        n_take = min(n_target, len(df_lbl))
        print(f"Class {label}: taking {n_take} out of {len(df_lbl)} jets")
        sel = df_lbl.sample(n=n_take, random_state=config["random_seed"])
        selected.append(sel)
    sel_df = pd.concat(selected, ignore_index=True)

    # Per-jet reweights (per-class)
    weights = np.zeros(len(sel_df))

    if config["reweight_mode"] == "none":
        weights[:] = 1.0

    elif config["reweight_mode"] == "flat_pt":
        for label in config["labels"]:
            mask = sel_df[config["branch_label"]] == label
            if mask.sum() > 0:
                weights[mask] = assign_weights(
                    sel_df.loc[mask, config["branch_pt"]],
                    hist[label]["pt_edges"],
                    hist[label]["pt_weights"]
                )

    elif config["reweight_mode"] == "flat_mass":
        for label in config["labels"]:
            mask = sel_df[config["branch_label"]] == label
            if mask.sum() > 0:
                weights[mask] = assign_weights(
                    sel_df.loc[mask, config["branch_mass"]],
                    hist[label]["mass_edges"],
                    hist[label]["mass_weights"]
                )

    elif config["reweight_mode"] == "flat_pt_mass_2d":
        for label in config["labels"]:
            mask = sel_df[config["branch_label"]] == label
            if mask.sum() > 0:
                weights[mask] = assign_2d_weights(
                    sel_df.loc[mask, config["branch_pt"]],
                    sel_df.loc[mask, config["branch_mass"]],
                    hist[label]["pt_edges"],
                    hist[label]["mass_edges"],
                    hist[label]["pt_mass_weights2d"]
                )
    
    print("Weighting done...")

    # Rebalance classes
    ref = config["reference_label"]
    class_sums = {l: weights[sel_df[config["branch_label"]] == l].sum() for l in config["labels"]}
    print("Class weight sums before rebalancing:", class_sums)
    if class_sums[ref] > 0:
        for l in config["labels"]:
            if class_sums[l] > 0:
                scale = class_sums[ref] / class_sums[l]
                weights[sel_df[config["branch_label"]] == l] *= scale
    
    class_sums = {l: weights[sel_df[config["branch_label"]] == l].sum() for l in config["labels"]}
    print("Class weight sums after rebalancing:", class_sums)

    print("Rebalancing done...")

    # Shuffle & split
    sel_df = sel_df.assign(weights=weights)
    train_df, test_df = train_test_split(sel_df, test_size=config["test_fraction"],
                                         random_state=config["random_seed"], shuffle=True)

    def save_split(df, name):
        out_dir = os.path.join(config["output_dir"], name)
        os.makedirs(out_dir, exist_ok=True)

        chunk = math.ceil(len(df) / config["n_splits_files"])

        for i in range(config["n_splits_files"]):
            print(f"Processing {name} split {i+1}/{config['n_splits_files']}...")
            part = df.iloc[i*chunk:(i+1)*chunk]
            if part.empty:
                continue

            print(f"  Number of jets: {len(part)}")

            graphs_out = []
            graph_class_counts = {label: 0 for label in config["labels"]}

            # Group by file_id so we load each graph file only once
            for fid, group in part.groupby("_file_id"):
                print(f"    Loading graphs from file_id {fid}, file_name {files_graphs[fid]}...")
                print(f"    Number of jets in this file: {len(group)}")
                locs = group["_local_index"].to_numpy()
                weights = group["weights"].to_numpy()

                # Load the graph file once
                graphs = torch.load(files_graphs[fid], map_location="cpu", weights_only=False)

                # Collect only the needed graphs
                for loc, w in zip(locs, weights):
                    g = graphs[int(loc)]
                    g.weight = torch.tensor([float(w)], dtype=torch.float32)
                    graphs_out.append(g)
                    graph_class_counts[int(g.y)] += 1

                # Free memory from this file
                del graphs
                # gc.collect()

            out_path = os.path.join(out_dir, f"{name}_part{i:03d}.pt")

            # Shuffle the graphs before saving
            np.random.seed(config["random_seed"])
            graph_classes = [int(g.y) for g in graphs_out]
            # print("Classes before shuffling:", graph_classes[:30])
            np.random.shuffle(graphs_out)
            graph_classes = [int(g.y) for g in graphs_out]
            # print("Classes after shuffling:", graph_classes[:30])
            torch.save(graphs_out, out_path)
            print(f"Saved {len(graphs_out)} graphs to {out_path}")

            print(f"Class distribution in this part: {graph_class_counts}")

    save_split(train_df, "train")
    # save_split_histograms(train_df, "train", config, weighted=False)
    save_split_histograms(train_df, "train", config, weighted=True)
    save_split_histograms(sel_df, "all", config, weighted=True)
    save_split(test_df, "test")

    print("Preprocessing done.")


if __name__ == "__main__":
    main()
