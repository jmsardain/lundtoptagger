import argparse
import os
import glob
import time
import gc

import uproot
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_multiclass import *
from tools.GNN_model_weight.utils_newdata import load_yaml, get_scores
from tools.utils_config import recursive_update, parse_dot_args


print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    parser.add_argument('--override', nargs='*', default=[], help='Overrides of the values in the config file in the form key.subkey=value')
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    # Override configuration with command line arguments
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    kT_selection = config['data']['kT_cut']
    filepath_placeholder_vals = dict(
        sample = config['data']['sample'],
        kT_cut = kT_selection
    )

    paths_to_test_file_root = config['data']['paths_to_test_file_root']
    if isinstance(paths_to_test_file_root, str):
        # paths_to_test_file_root can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        paths_to_test_file_root = [paths_to_test_file_root]
    files_root = []
    for file_path in paths_to_test_file_root:
        file_path = file_path.format(**filepath_placeholder_vals)
        files_root.extend(glob.glob(file_path))
    print ("paths_to_test_file_root:", paths_to_test_file_root)
    files_root = sorted(files_root)  # ensure the order is the same as files_graphs
    print ("files:", files_root)

    paths_to_test_file_graphs = config['data']['paths_to_test_file_graphs']
    if isinstance(paths_to_test_file_graphs, str):
        # paths_to_test_file_graphs can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        paths_to_test_file_graphs = [paths_to_test_file_graphs]
    files_graphs = []
    for file_path in paths_to_test_file_graphs:
        file_path = file_path.format(**filepath_placeholder_vals)
        files_graphs.extend(glob.glob(file_path))
    print ("paths_to_test_file_graphs:", paths_to_test_file_graphs)
    files_graphs = sorted(files_graphs)  # ensure the order is the same as files_root
    print ("files:", files_graphs)

    path_to_outdir = config['data']['path_to_outdir'].format(**filepath_placeholder_vals)
    os.makedirs(path_to_outdir, exist_ok=True)
    print("The output files will be saved to")
    print(path_to_outdir)

    # path_to_combined_ckpt = config['test']['path_to_combined_ckpt'][kT_selection]
    # print("ckpt used:", path_to_combined_ckpt )
    path_to_weights_dict = config['test']['path_to_combined_ckpt'] # I want to loop through the models, calculate scores, and save them

    output_suffix = config['data']['output_suffix'].format(**filepath_placeholder_vals)

    intreename = "FlatSubstructureJetTree"
    files_and_trees = {file_name: intreename for file_name in files_root}
    nentries_total = sum(entry[-1] for entry in uproot.num_entries(files_and_trees))
    nentries_done = 0

    batch_size = config['test']['batch_size']
    choose_model = config['test']['choose_model']

    t_filestart = time.time()

    # Set up model
    # TODO: test multiple models, so there is no need to re-load the data for each model
    if choose_model == "LundNet4Class":
        model = LundNet4Class()
    if choose_model == "LundNet":
        model = LundNet()
        # model = LundNet_old()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()
    if choose_model == "LundNet_plus_GN2X":
        model = LundNet_plus_GN2X()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Usually gpu 4 worked best, it had the most memory available
    print(f'\nUsing device: {device}')

    # model.load_state_dict(torch.load(path_to_combined_ckpt, map_location=device))
    # model.to(device)

    # Evaluation
    for file_number, (file_graphs, file_root) in enumerate(zip(files_graphs,files_root), start=1):
        t_start = time.time()

        # Load the data
        print(f"\nLoading file: {file_number}/{len(files_graphs)}\n", file_graphs)

        dataset = torch.load(file_graphs, weights_only=False)

        n_jets = len(dataset)
        print("Dataset size:", n_jets)
        delta_t_fileax = time.time() - t_start
        minutes, seconds = divmod(round(delta_t_fileax), 60)
        print(f"Time taken to load: {minutes:d} min {seconds:d} s")

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get the tree from the existing ROOT file,
        # either the file created by the Make_data.py script or from the scores file if it already exists
        print(f"Getting the tree from the existing ROOT file: {file_root}")
        filename_no_ext = os.path.splitext(os.path.basename(file_root))[0]  # get the input file name without the .root extension
        outfile_path = os.path.join(path_to_outdir, filename_no_ext) + output_suffix + ".root"
        outfile_path = outfile_path.format(**filepath_placeholder_vals)

        infile = outfile_path if os.path.exists(outfile_path) else file_root
        with uproot.open(infile) as f:
            arrays = f[intreename].arrays()

        

        # Predict scores
        print("\nCalculating scores...")
        for model_name, path_to_weights in path_to_weights_dict.items():
            model.load_state_dict(torch.load(path_to_weights, map_location=device))
            model.to(device)
            print(f'Using weights: {model_name} from {path_to_weights}')
            y_pred = get_scores_multi(test_loader, model, device)
            # Get the tagger_scores, scipping the first dummy entry

            tagger_scores = np.array(y_pred)  # for multi-class, the scores are in columns 0,1,2,3

            # Get number of classes
            num_classes = tagger_scores.shape[1]
            print(f"Number of classes: {num_classes}")

            # Store each class score in a separate branch
            for class_idx in range(num_classes):
                branch_name = config["test"]["scores_branch_name"].format(model=model_name) + f"_class{class_idx}"
                arrays[branch_name] = tagger_scores[:, class_idx]

            # # Add a new branch for the scores or overwrite the existing one
            # arrays[config["test"]["scores_branch_name"].format(model=model_name)] = tagger_scores
            del tagger_scores, y_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        delta_t_pred = time.time() - t_start - delta_t_fileax
        minutes, seconds = divmod(round(delta_t_pred), 60)
        print(f"Time taken to calculate predictions: {minutes:d} min {seconds:d} s")

        # Free up memory
        del dataset, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save the new scores to file
        # TODO: maybe this could be done more efficiently with PyROOT, without reading the whole tree and writing it again
        print("\nSaving scores to ROOT file...")
        with uproot.recreate(outfile_path) as f:
            f["FlatSubstructureJetTree"] = arrays
        print("Scores saved to:", outfile_path)

        # Free up memory
        del arrays
        gc.collect()

        # Time statistics
        delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
        minutes, seconds = divmod(round(delta_t_save), 60)
        print(f"Time taken to save: {minutes:d} min {seconds:d} s")

        nentries_done += n_jets
        time_per_entry = (time.time() - t_start)/(nentries_done)
        eta = time_per_entry * (nentries_total - nentries_done)
        minutes, seconds = divmod(round(eta), 60)
        print(f"\nEvaluated on {nentries_done} out of {nentries_total} jets")
        print(f"Estimated time until completion: {minutes:d} min {seconds:d} s")

    delta_t_total = time.time()-t_filestart
    minutes, seconds = divmod(round(delta_t_total), 60)
    print(f"\nTotal evaluation time: {minutes:d} min {seconds:d} s")


if __name__ == "__main__":
    main()