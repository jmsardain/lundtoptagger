import argparse
import csv
from datetime import datetime
import os

import numpy as np
import torch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import *
from plotting.utils_plots_matplotlib import hist_with_errors

import glob
import re
import sys

print("Libraries loaded!")


def objective(device, model, trial, train_ds, validation_ds, model_name, path_to_save ):
    lim = trial.suggest_int('lim', 2, 5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 1000, 6000)

    min_val = train_val(device, model, train_ds, validation_ds, learning_rate, batch_size, lim, model_name, path_to_save )

    return min_val


def train_val(device, model, train_ds, validation_ds, learning_rate, batch_size, lim, model_name, path_to_save ):

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=5*learning_rate)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    train_bgrej = []
    val_bgrej = []

    train_loss = []
    val_loss = []
    
    metrics_filename = path_to_save+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    min_val_loss = 999
    n_epochs = 20
    for epoch in range(n_epochs):
        train_loss.append(train_clas_2(train_loader, model, device, optimizer, optimizer2, optimizer3, epoch,lim))
        val_loss.append(my_test(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))

        model_dir = path_to_save + "models_lr{:}_lim{:}_bs{:}/".format(learning_rate, lim, batch_size)
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_filename = model_name + "_e{:03d}".format(epoch+1) + "_LossVal{:.5f}".format(val_loss[epoch]) + ".pt"
        torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
        
        if val_loss[epoch] < min_val_loss:
            min_val_loss = val_loss[epoch]
            path_minimun_val_loss = model_dir + model_filename
            print("path_minimun_val_loss:", path_minimun_val_loss)

    return min_val_loss, path_minimun_val_loss


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    add_arg('--ln_kT_cut', type=float, help="minimum value of kT kept for the training graphs")
    add_arg(
        '--do_combined_training',
        type=lambda x: str(x).lower(),
        choices=["true", "false", "yes", "no", "1", "0"],
        help="value can be true/false, yes/no, 0/1, case insensitive"
    )
    add_arg('--SA_step', type=int, help="Step of SA method")
    add_arg('--lr_part', type=int, help="learning rate selected in grid search loop")

    
    args = parser.parse_args()

    
    config_file = args.config
    config = load_yaml(config_file)
    ln_kT_cut = args.ln_kT_cut if args.ln_kT_cut is not None else config['data']['ln_kT_cut']
    do_combined_training = (
        True if args.do_combined_training in ["true", "yes", "1"] else
        False if args.do_combined_training in ["false", "no", "0"] else
        config['architecture']['do_combined_training']
    )
    
    # load the dataset
    path_to_file = config['data']['path_to_trainfiles']
    dataset = []
    path_to_file = path_to_file.format(ln_kT_cut=ln_kT_cut)
    print(path_to_file)
    path_to_file = glob.glob(path_to_file)
    print(path_to_file)
    
    if isinstance(path_to_file, str):
        # path_to_file can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        path_to_file = [path_to_file]
    for file_path in path_to_file:
        file_path = file_path.format(ln_kT_cut=ln_kT_cut)
        print("Loading file", file_path)
        #dataset += torch.load(file_path, weights_only=False) # weights_only=False added so that it works with PyTorch 2.6; it used to be the default
        dataset += torch.load(file_path)
        
    # apply jet mass and pT cuts
    if config['cut_pt_mass']:
        config_signal = load_yaml(config['config_signal_path'])[config['signal']]
        pt_range = config_signal['pt_range']
        mass_range = config_signal['mass_range']
        print("Filtering jets with pT in range", pt_range, "and mass in range", mass_range)
        dataset = [jet_graph for jet_graph in dataset
                   if  pt_range[0]   < jet_graph.pt   < pt_range[1]
                   and mass_range[0] < jet_graph.mass < mass_range[1]]

    # check the number of signal and background jets
    labels = np.array([jet_graph.y for jet_graph in dataset])
    num_signal = (labels==1).sum()
    num_background = (labels==0).sum()
    print("")
    print("Signal count:", num_signal)
    print("Background count:", num_background)

    # optionally flatten the mass and pt distributions and save plots of the distributions
    masses = np.array([jet_graph.mass for jet_graph in dataset])
    pts = np.array([jet_graph.pt for jet_graph in dataset])

    flatten_mass = config['flatten_mass']
    flatten_pt = config['flatten_pt']
    if flatten_mass and flatten_pt:
        weights_bkg = assign_2d_flat_weights_kde(masses[labels==0], pts[labels==0], bw_method='scott')
        weights_sig = assign_2d_flat_weights_kde(masses[labels==1], pts[labels==1], bw_method='scott')
    elif flatten_mass or flatten_pt:
        iterations  = config['num_iters']
        arrays_to_flatten_bkg = []
        arrays_to_flatten_sig = []
        n_bins = []
        if flatten_mass:
            arrays_to_flatten_bkg.append(masses[labels==0])
            arrays_to_flatten_sig.append(masses[labels==1])
            n_bins.append(config['n_bins_mass'])
        if flatten_pt:
            arrays_to_flatten_bkg.append(pts[labels==0])
            arrays_to_flatten_sig.append(pts[labels==1])
            n_bins.append(config['n_bins_pt'])
        weights_bkg = assign_flat_weights(*arrays_to_flatten_bkg, n_bins=n_bins, iterations=iterations)
        weights_sig = assign_flat_weights(*arrays_to_flatten_sig, n_bins=n_bins, iterations=iterations)
    else:
        weights_attr_name = 'fjet_weight_pt' if hasattr(dataset[0], 'fjet_weight_pt') else f'fjet_weight_pt_{config["signal"]}'
        weights_bkg = np.array([jet_graph[weights_attr_name] for jet_graph in dataset if jet_graph.y == 0], dtype=np.float64)
        weights_sig = np.array([jet_graph[weights_attr_name] for jet_graph in dataset if jet_graph.y == 1], dtype=np.float64)

    path_to_save = config['data']['path_to_save'].format(ln_kT_cut=ln_kT_cut)
    os.makedirs(path_to_save, exist_ok=True)
    print("\nResults will be saved to", path_to_save)

    for var_array, var_name, var_bins in zip([masses, pts], ['Mass', 'pT'], ['n_bins_mass', 'n_bins_pt']):
        hist_args = dict(
            bins = config[var_bins],
            density = True,
            fmt = "."
        )
        hist_with_errors(var_array[labels==0], label='Background', weights=weights_bkg, **hist_args, capsize=2)
        hist_with_errors(var_array[labels==1], label='Signal',     weights=weights_sig, **hist_args)
        plt.xlabel(f"LRJ {var_name} [GeV]")
        plt.ylabel('density')
        if var_name=="Mass" and not flatten_mass or var_name=="pT" and not flatten_pt:
            plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(os.path.join(path_to_save, f"{var_name}_distribution.png"))
        plt.close()
    
    for truth_label, label_name, weights_array in zip([0, 1], ['background', 'signal'], [weights_bkg, weights_sig]):
        hist_arrays = [masses[labels==truth_label], pts[labels==truth_label]]
        hist_args = dict(
            bins=(config['n_bins_mass'], config['n_bins_pt']),
            weights=weights_array,
            density=True
        )
        bin_counts_2d_hist = np.histogram2d(*hist_arrays, **hist_args)[0]
        min_bin_count = bin_counts_2d_hist[bin_counts_2d_hist > 0].min()
        print(f"Minimum bin count for {label_name}:", min_bin_count)

        plt.hist2d(*hist_arrays, **hist_args, cmin=min_bin_count)
        plt.colorbar(label='density')
        plt.xlabel('LRJ Mass [GeV]')
        plt.ylabel('LRJ pT [GeV]')
        plt.savefig(os.path.join(path_to_save, f"Mass_pT_distribution_{label_name}.png"))
        plt.close()

    print("Mass and pT plots saved")

    # rescale the weights so that the total weight of signal jets is equal to the total weight of background jets
    weights_signal_total = weights_sig.sum()
    weights_background_total = weights_bkg.sum()
    print("")
    print("Signal total weight:", weights_signal_total)
    print("Background total weight:", weights_background_total)
    scale_factor = weights_signal_total / weights_background_total
    print("Scale factor:", scale_factor)

    dataset_sig = [jet_graph for jet_graph in dataset if jet_graph.y == 1]
    dataset_bkg = [jet_graph for jet_graph in dataset if jet_graph.y == 0]

    for jet_graph, weight in zip(dataset_sig, weights_sig):
        jet_graph.weights = weight
    for jet_graph, weight in zip(dataset_bkg, weights_bkg):
        jet_graph.weights = weight*scale_factor

    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']

    dataset= shuffle(dataset_sig+dataset_bkg, random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))

    deg = torch.zeros(100, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
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




    ################################################

   
    #path_to_ckpt = config['retrain']['path_to_ckpt']

    if config['retrain']['flag']:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))

    if torch.cuda.is_available():
        device_id = 'cuda' if config['gpu'] is None else 'cuda:'+str(config['gpu'])
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print(f'\nUsing device: {device}')

        
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer_small = torch.optim.Adam(model.parameters(), lr=0.4*learning_rate)
    #optimizer2 = torch.optim.Adam(model.parameters(), lr=4*learning_rate)
    #optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name'].format(ln_kT_cut=ln_kT_cut)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    timestamp = datetime.now().strftime("%d%m-%H%M")
    metrics_filename = os.path.join(path_to_save, f"losses_{model_name}_{timestamp}.txt")


    #########################################################################################################################################
    
    # define here algorithm to be used; define learning_rate, batch_size and lim.
    min_val_loss_global = 999
    grid_search = config['hyperparameter']['SA_METHOD'] #True
    if grid_search:
        ### This methods perform successive grid search
        ### Each new succesive grid search reduced the amplitude of the hyperparameters explored
        lr_ini = config['hyperparameter']['lr_ini'] # 0.00035
        R_lr_ini = config['hyperparameter']['R_lr_ini'] # 0.0002
        n_lr = config['hyperparameter']['n_lr'] #3
        
        lim_range_ini = int(config['hyperparameter']['lim_range_ini']) #int(4)
        R_lim_range_ini = config['hyperparameter']['R_lim_range_ini'] #1.5
        n_lim_range = config['hyperparameter']['n_lim_range'] #3
        
        batch_ini = int(config['hyperparameter']['batch_ini']) #int(2500)
        R_batch_ini = config['hyperparameter']['R_batch_ini'] #1200
        n_batch = config['hyperparameter']['n_batch'] #4
        
        exp_lr        = config['hyperparameter']['exp_lr'] #2.5
        exp_lim_range = config['hyperparameter']['exp_lim_range'] #1.5	## only 2 first grid search will include variation in lim_range variable   
        exp_batch     =	config['hyperparameter']['exp_batch'] #2

    
        ## 3*3*4=36 #
        print("grid_search")

        #n_step = config['hyperparameter']['n_step'] ### this should be inside a loop, but probably is better to do it manually in different instances.
        n_step = args.SA_step if args.SA_step is not None else config['hyperparameter']['n_step']
        path_to_save_pre = path_to_save + "n_step_" + str(n_step-1)+'/**/*'
        path_to_save = path_to_save + "n_step_" + str(n_step)+'/'

        #lr_part = config['hyperparameter']['lr_part']
        lr_part = args.lr_part if args.lr_part is not None else config['hyperparameter']['lr_part']
        #add_arg('--SA_step', type=int, help="Step of SA method")
        #add_arg('--lr_part', type=int, help="learning rate selected in grid search loop")


        
        if n_step != 1 :
            ###### find minimal Validation loss in previous step. ######
            #print("path_to_save_pre", path_to_save_pre)
            list_models = glob.glob(path_to_save_pre)
            #print(list_models)
            min_ckpt = ""
            Vloss_ckpt_min = 999
            for ckpt in list_models:
                print(ckpt)
                Vloss_ckpt = re.search(r'LossVal(\d+\.?\d*).pt', ckpt)
                if float(Vloss_ckpt.group(1)) < Vloss_ckpt_min:
                    print("Vloss_ckpt",Vloss_ckpt.group(1))
                    ckpt_min = ckpt
                    Vloss_ckpt_min = float(Vloss_ckpt.group(1))
                    
                    lr_ini = float((re.search(r'models_lr([^_]+)_lim', ckpt)).group(1))
                    print("lr_ini", lr_ini)
                    lim_range_ini = int( float( (re.search(r'_lim(\d+\.?\d*)_bs', ckpt)).group(1)) ) 
                    print("lim_range_ini",lim_range_ini)
                    batch_ini = int(float((re.search(r'_bs(\d+\.?\d*)/Lund', ckpt)).group(1)) )
                    print("batch_ini",batch_ini)
                    
                    
        R_lr_ini = R_lr_ini / (exp_lr**(n_step-1) )
        #print("R_lr_ini", R_lr_ini)
        R_lim_range_ini = int( R_lim_range_ini / (exp_lim_range**(n_step-1))  )
        #print("R_lim_range_ini", R_lim_range_ini)
        R_batch_ini = int( R_batch_ini / exp_batch**(n_step-1)  )
        #print("R_batch_ini", R_batch_ini)
        
        if n_step > 2:  ## only 2 first grid search will include variation in lim_range variable ()
            n_lim_range = 1
        
        lr_range = np.linspace(lr_ini-R_lr_ini, lr_ini+R_lr_ini, num=n_lr) 
        print("lr_range:", lr_range)
        lim_range = np.linspace( lim_range_ini-R_lim_range_ini, lim_range_ini+R_lim_range_ini, num=n_lim_range) 
        print("lim_range:", lim_range)
        batch_size_range = np.linspace( batch_ini-R_batch_ini , batch_ini+R_batch_ini , num=n_batch ) 
        print("batch_size_range:", batch_size_range)

        lr_count = 0
        for learning_rate in lr_range:
            lr_count += 1
            #lr_part = config['hyperparameter']['lr_part'] 
            if lr_part > n_lr:
                print("learning rate range", lr_range)
                sys.exit("lr_part parameter in config file is out of range")
            if lr_part != 0:
                if lr_part != lr_count:
                    continue
            
            #print("learning_rate",learning_rate)
            for lim in lim_range:
                #print("lim",lim)
                for batch_size in batch_size_range:
                    batch_size = int(batch_size)
                    #print("batch_size",batch_size)
                    print("lr", learning_rate,"lim", lim,"bs", batch_size)
                    model = LundNet()
                    model.to(device)
                    min_val_loss_try, path_minimun_val_loss_try = train_val(device, model, train_ds, validation_ds, learning_rate, batch_size, lim, model_name, path_to_save )
        
                    if min_val_loss_try < min_val_loss_global:
                        min_val_loss_global = min_val_loss_try
                        path_minimun_val_loss_global = path_minimun_val_loss_try
                        print("path_minimun_val_loss_global:", path_minimun_val_loss_global)
    else:
        print("optuna")
        study = optuna.create_study(direction='minimize')  # Minimizar el validation loss
        study.optimize(lambda trial: objective(device, model, trial, train_ds, validation_ds, model_name, path_to_save ), n_trials=25)    
        print(f"Mejores parámetros encontrados: {study.best_params}")
        return study.best_params
    

if __name__ == "__main__":
    main()
