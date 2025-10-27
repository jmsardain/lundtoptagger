import argparse
import os
import glob
import time
from datetime import timedelta
import gc
from operator import itemgetter

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight, create_train_dataset_fulld_new_Ntrk_pt_weight_file
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    parser.add_argument('--override', nargs='*', default=[], help='Overrides in the form key.subkey=value')
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    # Override configuration with command line arguments
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    config_signal = load_yaml(config["signal_config_file"])
    signal = config["signal"]
    jet_label_branch = config["jet_label_branch"]
    jet_label_target = config_signal[signal]["signal_jet_label"]  # List of integers

    # Get some configuration parameters

    # Get the rootfile path
    rootfiles_placeholder_vals = dict(
        ntuple_tag = config["ntuple_tag"],
        jet_class = config["jet_class"],
    )
    path_to_files = config["path_to_rootfiles"].format(**rootfiles_placeholder_vals)
    files = glob.glob(path_to_files)[:config["n_files"]]

    intreename = "AnalysisTree"

    n_files = len(files)
    print(f"Processing {n_files} files")

    event_fractions = []
    event_factor = config["event_factor"] # Use factor less than 1 to reduce the number of total events
    for frac, n_chunks in config["event_fractions"].items():
        event_fractions.extend([frac] * n_chunks)
    if sum(event_fractions) > 1.0 + 1e-8:
        raise ValueError(f"Sum of event_fractions ({sum(event_fractions)}) exceeds 1.")

    # Select which event fractions to process
    if config["event_fraction_idx"] is not None:
        # Only process the specified fraction
        event_fraction_indices = [config["event_fraction_idx"]]
    else:
        # Process all fractions (default behavior)
        event_fraction_indices = list(range(len(event_fractions)))

    t_start = time.time()

    # Jet properties that will be loaded and saved in the output ROOT file
    # which will accompany the graphs file;
    # these properties have one numerical value per jet
    # If a property is not present in the input file, it will be skipped without an error
    jet_property_names = {      # keys are output branch names, values are input branch names
        "fjet_m":              "LRJ_mass",
        "fjet_pt":             "LRJ_pt",
        "fjet_eta":            "LRJ_eta",
        "fjet_phi":            "LRJ_phi",
        "fjet_truth_label":    "LRJ_truthLabel",
        "fjet_nProng_labels":  "LRJ_nprong", # Four-Prong labels
        "fjet_nQuark_labels":  "LRJ_CapturedQuarkCount",
        "fjet_Nconst_Charged": "LRJ_Nconst_Charged", # LRJ_Ntrk500, LRJ_Nconst?
        "GN2X_pqcd":           "GN2Xv01_pqcd",
        "GN2X_phbb":           "GN2Xv01_phbb",
        "GN2X_ptop":           "GN2Xv01_ptop",
        "GN2X_phcc":           "GN2Xv01_phcc",
        "fjet_tau42_wta":      "Tau42_wta", # Four-prong cut-based discriminant
    }
    # TODO: change this to just use the same names in the output file (requires modifying plotting code as well)
    
    # Additional variables which require some manipulation before they can be saved
    # because they need to be calculated or they have one value per event rather than per jet
    additional_output_vars = [
        "labels",                    # 1 for signal 0 for background
        "EventInfo_mcEventWeight",
        "EventInfo_mcChannelNumber", # dsid
    ]
    # add weights which make pT distribution flat
    # one or multiple variations depending on the configuration
    signals = [s for s in config_signal.keys() if s != "bkg_histos"] if signal=="all" else [signal]
    fjet_weight_pt_branches = [f"fjet_weight_pt_{s}" for s in signals] if signal=="all" or config["signal_name_in_weight"] else ["fjet_weight_pt"]
    additional_output_vars.extend(fjet_weight_pt_branches)

    # Calculate flat-pT weights, apply jet selection and kT cuts, and construct the graphs
    for frac_idx in event_fraction_indices:
        event_fraction = event_fractions[frac_idx] * event_factor
        print(f"\nProcessing event fraction {event_fraction} ({frac_idx}/{len(event_fractions)})")
        dataset = []
        primary_Lund_only_one_arr = []
        out_tree_dict = {branch_name: ak.Array([]) for branch_name in [*jet_property_names.keys(), *additional_output_vars]}
        for file_number, file in enumerate(files, start=1):
            print(f"\nLoading file: {file_number}/{n_files}\n", file)

            with uproot.open(file) as infile:
                tree = infile[intreename]

                dsids = tree["dsid"].array(library="np")
                dsid_test = dsids[0]                                 # check the first DSID, they should all be the same
                skip_dsids = set.intersection(*[set(config_signal[s]["skip_dsids"]) for s in signals])
                if dsid_test in skip_dsids: # don't lose time with jets that don't pass pt cut or wrong signal sample
                    print("Skipping file with DSID", dsid_test)
                    continue

                total_events = tree.num_entries

                # Calculate start and stop indices for this fraction
                prev_fractions = sum(event_fractions[:frac_idx])
                start_entry = int(total_events * prev_fractions)
                stop_entry = int(total_events * (prev_fractions + event_fraction))
                stop_entry = min(stop_entry, total_events)
                if start_entry >= stop_entry:
                    print(f"Skipping: start_entry {start_entry} >= stop_entry {stop_entry}")
                    continue
                print(f"Loading entries {start_entry}:{stop_entry} from {total_events} total entries")

                # Load the data
                jet_properties = {
                    jet_property: ak.flatten(tree[jet_property].array(entry_start=start_entry, entry_stop=stop_entry, library="ak"))
                    for jet_property in [*jet_property_names.values(), "jetLundZ", "jetLundKt", "jetLundDeltaR", "jetLundIDParent1", "jetLundIDParent2"]
                    if jet_property in tree
                }
                truth_labels_unflattened = tree["LRJ_truthLabel"].array(entry_start=start_entry, entry_stop=stop_entry, library="ak")
                numbers_of_jets_per_event = ak.num(truth_labels_unflattened)

                mcEventWeights = tree["mcEventWeight"].array(entry_start=start_entry, entry_stop=stop_entry, library="np")
                jet_properties["EventInfo_mcEventWeight"] = np.repeat(mcEventWeights, numbers_of_jets_per_event)       # expand out the array so it has same length as flattened array
                jet_properties["EventInfo_mcChannelNumber"] = np.repeat(dsids[start_entry:stop_entry], numbers_of_jets_per_event) # TODO: can I do this without numpy? expand out the array so it has same length as flattened array

                # Calculate flat-pT weights
                print("\nCalculating weights:")
                for fjet_weight_pt_branch, s in zip(fjet_weight_pt_branches, signals):
                    jet_properties[fjet_weight_pt_branch] = GetPtWeight(
                        jet_properties["LRJ_pt"],
                        jet_properties["LRJ_truthLabel"],
                        dsid_test,
                        config_signal[s],
                        SF=5,
                    )

                # TODO: just filter the arrays by mass, pT and eta before passing them to the dataset creation function

                # Construct the graphs, applying jet selection and kT cuts
                print("\nCreating PyTorch graphs:")
                passed_selection = []   # will be a boolean array, True if jet passes selection
                dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(
                    dataset,
                    *itemgetter("jetLundZ", "jetLundKt", "jetLundDeltaR", "jetLundIDParent1", "jetLundIDParent2")(jet_properties),
                    *itemgetter(jet_label_branch, "EventInfo_mcChannelNumber", "LRJ_Nconst_Charged", "LRJ_pt", "LRJ_mass", "LRJ_eta")(jet_properties),
                    weights = {fjet_weight_pt_branch: jet_properties[fjet_weight_pt_branch] for fjet_weight_pt_branch in fjet_weight_pt_branches},
                    GN2X_scores={
                        key: jet_properties[jet_property_names[key]]
                        for key in ["GN2X_pqcd", "GN2X_phbb", "GN2X_ptop", "GN2X_phcc"]
                        if jet_property_names[key] in jet_properties},
                    kT_selection=config["kT_cut"],
                    primary_Lund_only_one_arr=primary_Lund_only_one_arr,
                    passed_selection=passed_selection,
                    signal_jet_truth_labels=set().union(*[config_signal[s]["signal_jet_label"] for s in signals]),
                    signal_dsids=set().union(*[config_signal[s]["dsids"] for s in signals]),
                    pt_range=(
                        min(min(config_signal[s]["pt_range"]) for s in signals),
                        max(max(config_signal[s]["pt_range"]) for s in signals)
                    ),
                    mass_range= (
                        min(min(config_signal[s]["mass_range"]) for s in signals),
                        max(max(config_signal[s]["mass_range"]) for s in signals)
                    ),
                    eta_max=max(config_signal[s]["eta_max"] for s in signals),
                    min_splits=min(config_signal[s]["min_splits"] for s in signals),
                    include_pt=config["include_pt"],
                    binary_label=config["binary_label"],
                )

                for jet_property_out, jet_propety_in in jet_property_names.items():
                    if jet_propety_in in jet_properties:
                        out_tree_dict[jet_property_out] = ak.concatenate([out_tree_dict[jet_property_out], jet_properties[jet_propety_in][passed_selection]])
                    else:
                        print(f"Warning: {jet_propety_in} not found in file {file}, skipping")
                        if jet_property_out in out_tree_dict: del out_tree_dict[jet_property_out]
                for output_var in additional_output_vars:
                    if output_var!="labels":
                        out_tree_dict[output_var] = ak.concatenate([out_tree_dict[output_var], jet_properties[output_var][passed_selection]])

                gc.collect()

        out_tree_dict["labels"] = ak.Array([jet_graph.y for jet_graph in dataset])

        print("\nDataset created! len():", len(dataset))
        delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
        print(f"Time taken (hh:mm:ss): {delta_t_fileax}")

        # Save graphs and accompanying ROOT files
        filepath_placeholder_vals = dict(
            signal = config["signal"],
            jet_class = config["jet_class"],
            id = config["id"],
            kT_cut = config["kT_cut"],
            include_pt = "_with_pt" if config["include_pt"] else "",
            frac = f"_part{frac_idx}_{event_fraction*100:.2f}percent" if event_fraction < 1.0 else "",
        )
        out_dir = config["out_dir"].format(**filepath_placeholder_vals)
        os.makedirs(out_dir, exist_ok=True)

        out_file_name_graphs = config["out_file_name_graphs"].format(**filepath_placeholder_vals)
        output_path_graphs = os.path.join(out_dir, out_file_name_graphs)
        torch.save(dataset, output_path_graphs)
        print("Graphs saved to:", output_path_graphs)

        outfile_name_root = config["out_file_name_root"].format(**filepath_placeholder_vals)
        output_path_root = os.path.join(out_dir, outfile_name_root)
        with uproot.recreate(output_path_root) as outfile:
            outfile["FlatSubstructureJetTree"] = out_tree_dict
        print("Dataset written to ROOT file:", output_path_root)

if __name__ == "__main__":
    main()
