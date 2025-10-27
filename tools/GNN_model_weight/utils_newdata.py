import os
from typing import Union
import math

import yaml
import uproot
import awkward as ak
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from scipy.stats import entropy, gaussian_kde

from ..GNN_model_weight.models import mdn_loss, mdn_loss_new


def GetPtWeight(pts, truth_labels, dsid_input: int, signal_config: dict, SF: float = 5) -> np.array:
    """
    Return an array of weights for jets that make their pT distribution flat.

    Args:
        pts (array-like): Jet pT values.
        truth_labels (array-like): Integer large-R jet truth labels (e.g. 1 for tqqb, 2 for Wqq, 5 for Zqq, 10 for QCD).
        dsid_input (int): DSID of the input sample. It is assumed that all of the jets are from the same sample (or the same group of QCD samples).¸
        signal_config (dict): Dictionary containing the configuration for signal and background pT histograms.
        SF (float): Scale factor used for correct relative weighting of signal and background. Not important any more since weights are rescaled in training script to balance signal and background.
    Returns:
        np.array: An array of weights for the jets.
    """
    # get signal and backgound pT histograms
    filenames_bkg = signal_config["pt_hist_files_bkg"]["files"]
    histos_dir = signal_config["pt_hist_files_bkg"]["dir_path"]

    filename_Phythia = os.path.join(histos_dir, "mass_40-300_pt_200-3100/qcdP8.root")  # default file for background jets if no match found

    if dsid_input in signal_config["pt_hist_files_signal"]:
        filename_sig = signal_config["pt_hist_files_signal"][dsid_input]
    else:
        filename_sig = filename_Phythia
        print(f"WARNING: No signal histogram file found for DSID {dsid_input} for given signal configuration.")
        print("Instead, using default background histogram file for signal pT reweighting:", filename_sig)

    print("DSID:", dsid_input)
    # only works if all the data in a single ROOT file is from the same DSID
    if dsid_input not in signal_config["dsids"]:
        found_background_file = False
        for filename, dsid_range in filenames_bkg.items():
            if dsid_range[0] <= dsid_input <= dsid_range[1]:
                filename_bkg = os.path.join(histos_dir, filename)
                found_background_file = True
                break
        if not found_background_file:
            filename_bkg = filename_Phythia
            print(f"WARNING: No background histogram file found for DSID {dsid_input} for given signal configuration.")
    else:
        tmp_filename_list = list(filenames_bkg.keys())
        print('list(filenames_bkg.keys()):', tmp_filename_list)
        filename_bkg = os.path.join(histos_dir, tmp_filename_list[0])

    print("Using signal file:", filename_sig)
    print("Using background file:", filename_bkg)

    weights_file_sig = uproot.open(filename_sig)
    bin_counts_sig, bin_edges_sig = weights_file_sig["pt"].to_numpy()
    nbins_sig = len(bin_counts_sig)

    weights_file_bkg = uproot.open(filename_bkg)
    bin_counts_bkg, bin_edges_bkg = weights_file_bkg["pt"].to_numpy()
    n_bins_bkg = len(bin_counts_bkg)

    # calculate scaling factor between signal and background using Pythia sample histogram
    filename_reweight = filename_Phythia
    weights_file_reweight = uproot.open(filename_reweight)
    bin_counts_bkg_reweight, _ = weights_file_reweight["pt"].to_numpy()
    total_jets_qcd = np.sum(bin_counts_bkg_reweight)
    total_jets_signal = np.sum(bin_counts_sig)
    print("proportion QCD_pythia/SIGNAL", total_jets_qcd / total_jets_signal)
    #ERRORRR
    QCD_SIGNAL_proportion = total_jets_qcd / total_jets_signal
    sig_bkg_proportion = SF #5  ## if is taked 5% of signal and 1% of qcd for training then sig_bkg_proportion=5
    scale_factor = (n_bins_bkg/nbins_sig) / sig_bkg_proportion #1
    scale_factor = scale_factor * QCD_SIGNAL_proportion
    print("scale factor:", scale_factor)

    # calculate the weight for each pT bin as the inverse of the bin count, with some scale factors
    Inv_hist_bg = np.where(
        bin_counts_bkg==0, 0,                                   # if bin count is zero, set weight to zero
        np.sum(bin_counts_bkg) / (n_bins_bkg * bin_counts_bkg)  # otherwise, calculate the inverse weight
    )
    Inv_hist_sig = np.where(
        bin_counts_sig==0, 0,
        np.sum(bin_counts_sig) / (nbins_sig * bin_counts_sig) * scale_factor
    )

    # for each jet pT, get the index of the corresponding pT bin
    pt_bin_indices_sig = np.digitize(pts, bin_edges_sig) - 1          # -1 to get zero-based indices
    pt_bin_indices_sig = np.clip(pt_bin_indices_sig, 0, nbins_sig-1)  # for bin indices above the last, change them to the index of the last bin
    pt_bin_indices_bkg = np.digitize(pts, bin_edges_bkg) - 1
    pt_bin_indices_bkg = np.clip(pt_bin_indices_bkg, 0, n_bins_bkg-1)

    # assign weight based on which pT bin the jet pT falls into and whether it is signal or background
    weights_out = np.where(
        truth_labels == 10,
        Inv_hist_bg[pt_bin_indices_bkg],
        Inv_hist_sig[pt_bin_indices_sig] # jets with label other than 10 are reweighted by the signal pT histogram, some of these are actually not signal and are removed later
    )

    return weights_out


def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def assign_flat_weights(*arrays, n_bins=50, iterations=2):
    """
    Assign weights to array entries so that the marginal distributions of the input arrays become flat.

    The procedure works by iteratively normalizing the total weight in each bin to equal 
    the average total weight of all (nonempty) bins.

    Parameters:
      *arrays      : one or more array-like inputs to reweight (e.g., mass, pT). They should have the same lengths.
      n_bins       : int or list of ints, the number of bins to use for reweighting each array.
                     If a single int is provided, it is used for all arrays.
      iterations   : int, number of iterations of reweighting.

    Returns:
      weights      : NumPy array of weights for each entry.
    """
    arrays = [np.asarray(arr) for arr in arrays]
    weights = np.ones_like(arrays[0], dtype=float)

    # Ensure n_bins is a list with the same length as arrays
    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(arrays)
    elif len(n_bins) != len(arrays):
        raise ValueError("n_bins must be an int or a list with the same length as the number of input arrays.")

    # Iteratively flatten the distributions
    for _ in range(iterations):
        for arr, bins in zip(arrays, n_bins):
            # Define bin edges for the current array
            bin_edges = np.linspace(arr.min(), arr.max(), bins + 1)
            # Find the bin index for each event
            bin_indices = np.digitize(arr, bin_edges) - 1  # subtract 1 because digitize is 1-indexed
            # Calculate the sum of weights in each bin
            bin_sums = np.array([weights[bin_indices == i].sum() for i in range(bins)])
            # We want all non-empty bins to have the same total weight
            nonzero = bin_sums > 0
            target = bin_sums[nonzero].mean() if nonzero.any() else 1.0
            
            # Rescale the weights in each bin
            for i in range(bins):
                if bin_sums[i] > 0:
                    idx = (bin_indices == i)
                    weights[idx] *= (target / bin_sums[i])

    return weights

def assign_2d_flat_weights_kde(mass, pt, bw_method='scott', eps=1e-9):
    """
    Assign weights to events such that the 2D (mass, pT) distribution becomes flat.
    
    This function estimates the joint density using a kernel density estimator (KDE)
    and assigns weights inversely proportional to the local density.
    
    Parameters:
      mass       : array-like, mass values for each event.
      pt         : array-like, pT values for each event.
      bw_method  : str or scalar, the method or factor used to calculate the KDE bandwidth.
                   ('scott' or 'silverman' are common choices, or you can provide a scalar)
      eps        : float, a small number used to prevent division by zero.
    
    Returns:
      weights    : NumPy array of weights for each event.
    """
    # Combine mass and pT into a 2xN array for the KDE
    data = np.vstack([mass, pt])
    
    # Instantiate and evaluate KDE on the data points
    kde = gaussian_kde(data, bw_method=bw_method)
    density = kde.evaluate(data)
    
    # Compute weights as the inverse density. The eps prevents division by zero.
    weights = 1.0 / np.maximum(density, eps)
    
    # Normalize weights so that the average weight is 1 (optional, but useful for stability)
    weights /= np.mean(weights)
    
    return weights

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def create_train_dataset_fulld_new_Ntrk_pt_weight_file(
    graphs: list[Data],
    z, k, d, edge1, edge2, label, dsids, Ntracks, jet_pts, jet_ms, jet_etas,
    weights: dict[str, ak.Array],
    GN2X_scores,
    kT_selection: Union[float, None],
    primary_Lund_only_one_arr: list,
    passed_selection: list[bool],
    signal_jet_truth_labels: list[int],
    signal_dsids: list[int],
    pt_range: tuple = (350, 3200),
    mass_range: tuple = (0, float('inf')),
    eta_max: float = 2.0,
    min_splits: int = 3,
    include_pt: bool = False,
    binary_label: bool = False,
) -> list[Data]:
    """
    Create a list of graphs for tagging.

    Args:
        graphs (list[Data]): List to which the generated torch_geometric.data.Data objects will be appended.
        z (array): 2D array, with an array of z values for each jet.
        k (array): 2D array, with an array of kT values for each jet.
        d (array): 2D array, with an array of ΔR values for each jet.
        edge1 (array): Array of edge1 values.
        edge2 (array): Array of edge2 values.
        label (array): Array of jet truth labels (1 for top, 2 for W, 10 for QCD).
        dsids (array): Array of DSIDs for the jets.
        Ntracks (array): Array of Ntracks values.
        jet_pts (array): Array of jet pT values.
        jet_ms (array): Array of jet mass values.
        jet_etas (array): Array of jet pseudorapidity values.
        weights (dict[str, ak.Array]): Dictionary of arrays with jet weights.
        GN2X_scores (dict[str, array]): Dictionary of arrays with GN2X scores for the jets.
        kT_selection (float | None): kT selection threshold.
        primary_Lund_only_one_arr (list): List to keep track of how many jets have only 1 splitting.
        passed_selection (list): List to keep track of jets that passed the selection criteria.
        signal_jet_truth_labels (list[int]): List of jet truth labels that are treated as signal (e.g., [1] for top, [2] for W).
        signal_dsid (int): List of DSIDs that signal jets are taken from.
        pt_range (tuple): Minimum and maximum jet pT values for selected jets, in GeV.
        mass_range (tuple): Minimum and maximum jet mass values for selected jets, in GeV.
        eta_max (float): Maximum absolute value of jet pseudorapidity, for selected jets.
        min_splits (int): Minimum number of splittings, or emissions, for a jet to be selected.
        include_pt (bool): Whether to include pT as a graph attribute.
        binary_label (bool): Whether to use binary labels (1 for signal, 0 for background) instead of original labels.

    Returns:
        list[Data]: List of torch_geometric.data.Data objects.
    """
    buildID_from_graphs = 0
    Primary_Lund_Plane = 0
    extra_node = 0

    # loop over jets
    for i in trange(len(z), miniters=len(z) // 10, maxinterval=60*60*2, desc="Processing jets, printing at min. 10% intervals or every 2 hours"):
        '''
        label_np = ak.to_numpy(label[i])
        jet_pts_np = ak.to_numpy(jet_pts[i])
        jet_ms_np = ak.to_numpy(jet_ms[i])
        label_np = label_np.astype(float)
        jet_pts_np = jet_pts_np.astype(float)
        jet_ms_np = jet_ms_np.astype(float)
        '''

        # skip jets with mass, pT or eta outside the specified ranges
        # or with less than the specified number of splittings
        if (not (pt_range[0] < jet_pts[i] < pt_range[1])
            or not (mass_range[0] < jet_ms[i] < mass_range[1])
            or not (abs(jet_etas[i]) < eta_max)
            or len(z[i]) < min_splits
            # skip jets which are not signal (1 for top and 2 for W) or background (10)
            or dsids[i] in signal_dsids and label[i] not in signal_jet_truth_labels) or (dsids[i] not in signal_dsids and label[i]!=10
        ):
            passed_selection.append(False)
            continue
        else:
            passed_selection.append(True)  # changed to False later for some conditions

        if binary_label:
        # label signal as 1 and background as 0
            if label[i] == 10:
                label_out = 0
            if label[i] in signal_jet_truth_labels:
                label_out = 1
        else:
            label_out = label[i] # Dump the raw labels

        # convert LJP variables to appropriate format
        z_out = ak.to_numpy(z[i])
        k_out = ak.to_numpy(k[i])
        d_out = ak.to_numpy(d[i])
        
        z_out += 1e-4 
        k_out += 1e-4 
        d_out += 1e-4 
        
        z_out = np.log(1/z_out)
        k_out = np.log(k_out)
        d_out = np.log(1/d_out)
        
        
        ## lets go to do kt cut; to do this first we need to recover parentID1 and parentID2 (the ones that have a lot of -1) 
        if buildID_from_graphs==1:
            edges1 = ak.to_numpy(edge1[i]) ## it's not necesary edge2[i], it has the same information
            #print(len(edges1)/2)
            len_edges = int(len(edges1)/2)
            edges_A = edges1[:len_edges] # sons
            edges_B = edges1[len_edges:] # parents ; then edges_B[i] > edges_A[i]
            '''
            for x in range(len(edges1)):
                print(edges1[x])
            '''
            id1_id2_edge = 0
            '''
            print("edges_A len()->",len(edges_A))
            print("edges_B len()->",len(edges_B))
            print("edges_A",edges_A)
            print("edges_B",edges_B)
            '''
            for j in range(0,len(edges_A)):
                if j == len(edges_A)-1:
                    id1_id2_edge = j + 1
                    break
                if edges_A[j+1] < edges_A[j]:
                    id1_id2_edge = j + 1
                    break
            
            edges_A_1 = edges_A[id1_id2_edge:] 
            edges_A_2 = edges_A[:id1_id2_edge] 
            edges_B_1 = edges_B[id1_id2_edge:] 
            edges_B_2 = edges_B[:id1_id2_edge]
            '''
            print("edges_A_1",edges_A_1)
            print("edges_A_2",edges_A_2)
            print("edges_B_1",edges_B_1)
            print("edges_B_2",edges_B_2)
            '''
            ## it's time to recover parentID1 (using edges_A_1 and edges_B_1) and parentID2
            parentID1 = []
            parentID2 = []
            for j in range (0,len(z[i]) ):
                if len(edges_B_1) == 0:
                    parentID1.append(-1)
                elif j == edges_A_1[0]:
                    parentID1.append(edges_B_1[0])
                    edges_A_1 = np.delete(edges_A_1,0)
                    edges_B_1 = np.delete(edges_B_1,0)
                else:
                    parentID1.append(-1)
                    
                if len(edges_B_2) == 0:
                    parentID2.append(-1)
                elif j == edges_A_2[0]:
                    parentID2.append(edges_B_2[0])
                    edges_A_2 = np.delete(edges_A_2,0)
                    edges_B_2 = np.delete(edges_B_2,0)
                else:
                    parentID2.append(-1)
            
            ## Now using parentID1 and parentID1 let's go and do kT cut 
            ## I found both parentID because I think in this way code run faster, I don't want to do 
            ## extra loops or complex functions in a data sample with millions of graphs
            ### previous steps can be deleted if we take parentID1 and parentID2 from previous code
            #print("ID1   :",parentID1)
            #print("ID2   :",parentID2)
    
            
            ## here ID2 is the HARDEST branch!!

            # this fix should be not necessary anymore
            for j in range(0, len(parentID1)):
                if parentID1[j] == j :
                    #print("warning!")
                    parentID1[j] = -1
                if parentID2[j] == j :
                    #print("warning!")
                    parentID2[j] = -1
                    
        ## I just don't want to change some lines, this mix between 1 and 2 should be remove in next version
        if buildID_from_graphs != 1:
            parentID1 = ak.to_numpy(edge2[i]) #edge2
            parentID2 = ak.to_numpy(edge1[i]) #edge1
        
        # python3 weight_class_train-Copy1.py configs/config_class_train_top.yaml        
        index_count = []
        selected_nodes = []
        index_count_out = []
        kT_Cut = kT_selection if kT_selection is not None else -np.inf # 0.0 , 0.4 0.9, 2, 2.8 
        nodes_pass_KT = []
        node_kt_step = 0 ## used to renamed edges properly ()
        node_index = 0
        prev_cur_index = 0

        '''
        if i!=1061:
            continue
        print("i",i)
        print("edges_A",edges_A)
        print("edges_B",edges_B)
        print("parentID1",parentID1)
        print("parentID2",parentID2)
        print("k_out[0]",k_out[0])
        '''
        
        nodes_selected = []
        if Primary_Lund_Plane == 1:
            #print("ONLY PRIMARY LUND WILL BE USED!")
            nodes_primary_count = 0
            for j in range(0 , len(z[i])):
                if nodes_primary_count==0:  #j == 0 :
                    #j_ID1_next = parentID1[j]
                    j_ID1_next = parentID2[j]
                #selected_nodes = []
                #if k_out[j] <= kT_Cut : 
                if (k_out[j] <= kT_Cut): #  or j==0 or (j in parentID1) : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                if nodes_primary_count>0 and j != j_ID1_next: # j>0
                    #print("222222")
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                nodes_selected.append(True)
                nodes_primary_count +=1;
                #j_ID1_next = parentID1[j]
                j_ID1_next = parentID2[j]
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    #index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
                    '''
                    if len(graphs)==520:
                        #print("k_out[0]:", k_out[0], "  k_out[1]:", k_out[1])
                        print("cur_index:", cur_index)
                        print("index_1:", index_1, "kt(index_1)", k_out[index_1])
                        print("index_2:", index_2, "kt(index_2)", k_out[index_2])
                    '''
                    '''
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                            #if len(graphs)==520:
                            #    print("len(selected_nodes)inside  1:", len(selected_nodes))
                            #    print("len(index_count_out)inside 1:", len(index_count_out))
                        else:
                            index_count.append(index_1)
                    '''
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1                             
                        else:
                            index_count.append(index_2)
                    #'''
        ######################################################################################
        else:
            for j in range(0 , len(z[i])):
                #index_count.append(j) # this line here is an error!
                #selected_nodes = []
                if k_out[j] <= kT_Cut : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    continue
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
    
                    '''
                    if len(graphs)==520:
                        #print("k_out[0]:", k_out[0], "  k_out[1]:", k_out[1])
                        print("cur_index:", cur_index)
                        print("index_1:", index_1, "kt(index_1)", k_out[index_1])
                        print("index_2:", index_2, "kt(index_2)", k_out[index_2])
                    '''
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                            '''
                            if len(graphs)==520:
                                print("len(selected_nodes)inside  1:", len(selected_nodes))
                                print("len(index_count_out)inside 1:", len(index_count_out))
                            '''
                        else:
                            index_count.append(index_1)
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1 
                            '''
                            if len(graphs)==520:
                                print("len(selected_nodes)inside  2:", len(selected_nodes))
                                print("len(index_count_out)inside 2:", len(index_count_out))
                            '''
                        else:
                            index_count.append(index_2)
        
        ##transform edges numeration and avoid isolated nodes 
        '''
        print("nodes before kT slection: ", len(k_out) )
        print("nodes after kT slection: ", len(k_out[k_out > kT_Cut]) )
        print("len(index_count_out)",len(index_count_out))
        print("len(selected_nodes)",len(selected_nodes))
        #print("len(nodes_pass_KT)",len(nodes_pass_KT))
        '''
        if len(k_out[k_out > kT_Cut]) < 1:
            passed_selection[i] = False
            continue
        
        #print("index_count_out  :",index_count_out)
        #print("selected_nodes   :",selected_nodes)

        for j in range(0,len(index_count_out)):
            ## 1+ in order to add an extra node
            if extra_node==1:
                index_count_out[j] = int(1 + index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int(1 + selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
            else:
                index_count_out[j] = int( index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int( selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
        if extra_node==1:
            index_count_out.insert(0,0)
            selected_nodes.insert(0,1)
        
        
        #print("index_count_out Af:",index_count_out)
        #print("selected_nodes Af:",selected_nodes)
        #if i > 264:
        #    break
        #print(len(j))
            
        index_count_out = np.array(index_count_out, dtype=int )
        selected_nodes = np.array(selected_nodes, dtype=int)
        #index_count_out = index_count_out.astype(int)
        #selected_nodes = selected_nodes.astype(int)
        
        #print("1", selected_nodes)
        #print("1.5", selected_nodes[1])
        #print("2", type(selected_nodes[1]) )

        ## kt mask for feature
        if Primary_Lund_Plane==1:
            k_mask = np.array(nodes_selected)
        if Primary_Lund_Plane==0:
            k_mask = k_out > kT_Cut
        z_out = z_out[k_mask]
        k_out = k_out[k_mask]
        d_out = d_out[k_mask]
        
        
        mean_z, std_z = 2.0568479032747313, 1.4450598054504056
        mean_dr, std_dr = 3.8597358364389427, 2.2748462855901073
        mean_kt, std_kt = -2.379904791478249, 2.940813577366582
        #mean_ntrks, std_ntrks = 26.556999184747827, 16.53733685428723 #only qcd good partition
        #mean_ntrks, std_ntrks = 39.81133623360089, 10.99193693271175
        mean_ntrks, std_ntrks = 57.588158609500134, 23.900100132781983
        
        z_out = (z_out - mean_z) / std_z
        k_out = (k_out - mean_kt) / std_kt
        d_out = (d_out - mean_dr) / std_dr
        Ntrk = (Ntracks[i] - mean_ntrks) / std_ntrks

        #print("3", z_out)
        #print("3.5", z_out[1])
        #print("4", type(z_out[1]) )

        z_out = z_out.astype(float)
        k_out = k_out.astype(float)
        d_out = d_out.astype(float)

        #edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        edge_ID1 = np.concatenate((index_count_out, selected_nodes))
        edge_ID2 = np.concatenate((selected_nodes, index_count_out))
        edge = torch.tensor(np.array([edge_ID1, edge_ID2]) , dtype=torch.int64)
        #edge = np.array([edge_ID1, edge_ID2]).astype(int)
        

        vec = []
        ## in order to add an extra node
        if extra_node==1:
            #print(index_count_out)
            #print(d_out)
            d_out = np.append(0, d_out)
            z_out = np.append(0, z_out)
            k_out = np.append(0, k_out)

        vec.append(np.array([d_out, z_out, k_out]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        vec=torch.tensor(vec, dtype=torch.float).detach()

        graph_size = 1
        if len(k_out) == 1:
            primary_Lund_only_one_arr.append(1)
            #continue
            #edge = torch.tensor([[0,0], [0,0]], dtype=torch.int64)
            #edge = torch.tensor([[0], [0]], dtype=torch.int64)
            edge = torch.tensor([[], []], dtype=torch.int64)
            vec = torch.unsqueeze(vec, dim=0)
            graph_size = 0

        '''
        if len(k_out) == 2 and len(graphs)==520:
            print("graph number:", len(graphs))
            print("ID1_f:",parentID1)
            print("ID2_f:",parentID2)
            print("edge_index", edge)
        '''

        #print("5", edge)
        #print("5.3", edge[0,1])
        #print("5.8", edge[0].dtype )
        #print("6", edge[0,1].dtype )

        #print("weights", weight[i])
        #print("weights type:", type(weight[i]) )
        #print("pt", jet_pts[i])
        #print("mass", jet_ms[i])
        #print("mass type:", type(jet_ms[i]) ) # <class 'numpy.float64'>

        
        if len(edge_ID1)<1:
            primary_Lund_only_one_arr.append(1)
            #print("k_out",k_out , "  edge_ID1:", edge_ID1)
            passed_selection[i] = False
            continue
            #print("x",vec)
            #print("edge",edge)
        
        #print("edge",edge)
        #print("edge1",edge[0])
        #print("edge2",edge[1])

        graph = Data(
            x = vec.detach(),
            #edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
            edge_index = edge.detach(),
            Ntrk = torch.tensor(Ntrk, dtype=torch.float).detach(),
            #graph_size = torch.tensor(graph_size, dtype=torch.float).detach(),
            mass =  float(jet_ms[i]), #torch.tensor(jet_ms[i], dtype=torch.float).detach(),
            y = float(label_out), #torch.tensor(label_out, dtype=torch.float).detach() ))
        )
        if include_pt:
            graph["pt"] = float(jet_pts[i]) #torch.tensor(jet_pts[i] , dtype=torch.float).detach()
        for weight_name, weights_array in weights.items():
            graph[weight_name] = float(weights_array[i])
        for score_name, scores in GN2X_scores.items():
            graph[score_name] = float(scores[i])

        graphs.append(graph)
        '''
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
                           #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           Ntrk=torch.tensor(Ntrk, dtype=torch.float).detach(),
                           weights =torch.tensor(weight[i], dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label_out, dtype=torch.float).detach() ))
        '''
        #print(graphs[-1])
        #print(graphs[-1].x)
        #print(graphs[-1].edge_index)
        '''
        if len(k_out) == 1:
            print("1111111111")
            print(graphs[-1])
        if len(k_out) == 2 and len(graphs)%2==0 :
            print("2222222222")
            print(graphs[-1])
        '''

    print("all_graphs_count_graphs:", len(graphs))
    print("primary_Lund_only_one:", np.sum(primary_Lund_only_one_arr))
    print("percent_graphs:", 1 - np.sum(primary_Lund_only_one_arr) / len(graphs) )
        
    return graphs


def train(loader, model, device, optimizer):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        l2_lambda = 0.01 # regularization strength
        for param in model.parameters():
            if param.dim() > 1:
                # apply L2 regularization to all parameters except biases
                loss = loss + l2_lambda * nn.MSELoss()(param, torch.zeros_like(param))

        loss.backward()

        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def train_clas(loader, model, device, optimizer1, optimizer2, optimizer3, epoch):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        if len(data)<1024:
            continue
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        if epoch < 8:
            optimizer3.step()
        elif epoch < 16:
            optimizer2.step()
        else:
            optimizer1.step()
    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test(loader, model, device):
    model.eval()
    #print("init my_test()")
    #time.sleep(600)
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        data = data.to(device)
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y, weight=new_w)
        loss_all += data.num_graphs * loss.item()
    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader, model, device):
    model.eval()
    total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        # print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]

#### include adversarial and combined training
def train_adversary_2(loader, clsf, adv, optimizer, device, loss_parameter, loss_weights):
    clsf.eval()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0
    
    for data in loader:
        clsf.eval()
        if len(data)<512:
            continue
        batch_counter+=1
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1)) 
        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)

        #print(new_pt[:2], " new_pt  " , torch.log(new_pt[:2]) )
        mask_bkg = new_y.lt(0.5)
        optimizer.zero_grad()
        cl_out = clsf(cl_data)
        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)
        
        #adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1) ), torch.reshape(cl_data.pt[mask_bkg], (int(list(cl_data.pt[mask_bkg].shape)[0]),1) ) ) , 1)
        
        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)

        #adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(cl_data.pt[mask_bkg], (len(cl_data.pt[mask_bkg]),1) )   )  ,1)

        pi, sigma, mu = adv(adv_inp)
        
        #print("batch_counter",batch_counter)
        '''
        print("---------------------------------------")
        print( torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) )   )
        print("---------------------------------------")
        print("mu size->", mu.size(), "   pi size->",pi.size() ,"   sigma size->", sigma.size()  )
        print(mu[0])
        print("---------------------------------------")        
        print(pi[0])
        print("---------------------------------------")
        print(sigma[0])
        print("---------------------------------------")
        #'''
        #loss2 = loss_weights[1] * mdn_loss(pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        #loss2 = loss_weights[1] * loss_parameter * mdn_loss_new(pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        #loss2 = loss_weights[1] * mdn_loss_new(pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        loss2 = loss_weights[1] * mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])

        #print("loss_adv->",loss2.item())
        
        loss2.backward()
        loss = loss_weights[1] * loss1 + loss_parameter*loss2
        
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer.step()
        
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


def test_combined(loader, clsf, adv, device, loss_parameter, loss_weights ):
    clsf.eval()
    adv.eval()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    for data in loader:
        if len(data)<512:
            continue
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        cl_out = clsf(cl_data)
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1))

        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0
        
        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)

        #adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(cl_data.pt[mask_bkg], (len(cl_data.pt[mask_bkg]), 1))), 1)
        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)
        
        pi, sigma, mu = adv(adv_inp)

        loss2 = mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        
        loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2
        loss_clsf += loss_weights[0] * cl_data.num_graphs * loss1.item()
        loss_adv += loss_weights[1] * cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        #print("loss_adv->",loss_adv)
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)



def train_combined_2(loader, clsf, adv, optimizer_cl, optimizer_adv, device, loss_parameter, loss_weights):
    clsf.train()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0
    jsd_total = 0

    for data in loader:
        batch_counter+=1
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1))

        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)
        
        mask_bkg = new_y.lt(0.5)
        optimizer_cl.zero_grad()
        optimizer_adv.zero_grad()
        cl_out = clsf(cl_data)

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0

        #adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)
        pi, sigma, mu = adv(adv_inp)
        
        #print("pi[:2]---------------------------------------")
        #print(pi[:2])
        '''
        print("---------------------------------------")
        #print( torch.reshape(new_mass[mask_bkg], (len(new_pt[mask_bkg]),1) )   )
        print("---------------------------------------")
        print("mu size->", mu.size(), "   pi size->",pi.size() ,"   sigma size->", sigma.size()  )
        print("mu---------------------------------------")
        print(mu[:2])
        print("pi---------------------------------------")
        print(pi[:2])
        print("sigma---------------------------------------")
        print(sigma[:2])
        print("---------------------------------------")
        '''
        #print(len(loader.dataset))
        
        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)
        #loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)),new_w[mask_bkg])
        #loss2 = mdn_loss_new(pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        loss2 = mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        
        loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2
        loss.backward()
    
        loss_clsf += loss_weights[0] * cl_data.num_graphs * loss1.item()
        loss_adv += loss_weights[1] * cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step() 
        optimizer_adv.step()
        
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


def aux_metrics(loader, clsf, adv, device, MASSBINS):
    clsf.eval()
    adv.eval()
    counter = 0
    bkg_tagged = 0
    bkg_total = 0
    jsd_total = 0
    nans = 0
    jsd_counter = 0
    mass_tagged = np.array([])
    mass_untagged = np.array([])
    for data in loader:
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        #    print ("true labels",new_y)
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        cl_out = clsf(cl_data)
        mask_tag = cl_out.lt(0.5)
        mask_untag = cl_out.ge(0.5)

        bkg_tagged+=torch.count_nonzero(mask_untag&mask_bkg)
        bkg_total+=torch.count_nonzero(mask_bkg)

        p, _ = np.histogram(np.array(new_mass[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
        f, _ = np.histogram(np.array(new_mass[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)

        jsd = JSD(p,f)
        if math.isnan(jsd):
            nans+=1
        else:
            jsd_total +=jsd
            jsd_counter+=1
  #      print ("jsd",jsd)
    if bkg_tagged:
        eff = bkg_total/bkg_tagged
    else:
        eff = bkg_total*0

    if jsd_counter:
        jsd_total = jsd_total/jsd_counter
    else:
        jsd_total = 0
    return float(eff.cpu()), jsd_total

def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))

def train_clas_2(loader, model, device, optimizer1, optimizer2, optimizer3, epoch, lim):
    #print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        if len(data)<512:
            continue
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        if epoch < lim:
            optimizer3.step()
        elif epoch < 2*lim:
            optimizer2.step()
        else:
            optimizer1.step()
    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)
