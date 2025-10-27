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
from tools.GNN_model_weight.utils_multiclass import *
from plotting.utils_plots_matplotlib import hist_with_errors

print("Libraries loaded!")


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()

    config_file = args.config
    config = load_yaml(config_file)
    
    # load the dataset
    path_to_trainfiles = config['data']['path_to_trainfiles']
    path_to_validationfiles = config['data']['path_to_validationfiles']

    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']

    # define custom dataset
    train_ds = GraphIterableDataset(data_dir=path_to_trainfiles, limit_files=20)
    validation_ds = GraphIterableDataset(data_dir=path_to_validationfiles, limit_files=20)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])


    print ("train dataset size:", len(train_ds))
    print ("class counts in training dataset:", train_ds.get_class_counts())
    print ("class weights in training dataset:", train_ds.get_class_weights())
    print ("validation dataset size:", len(validation_ds))
    print ("class counts in validation dataset:", validation_ds.get_class_counts())
    print ("class weights in validation dataset:", validation_ds.get_class_weights())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet4Class":
        model = LundNet4Class()
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

    if torch.cuda.is_available():
        device_id = 'cuda' if config['gpu'] is None else 'cuda:'+str(config['gpu'])
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print(f'\nUsing device: {device}')

    #model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name']
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    timestamp = datetime.now().strftime("%d%m-%H%M")
    path_to_save = config['data']['path_to_save']
    os.makedirs(path_to_save, exist_ok=True)
    print("\nResults will be saved to", path_to_save)
    metrics_filename = os.path.join(path_to_save, f"losses_{model_name}_{timestamp}.txt")

    for epoch in range(n_epochs):
        train_loss.append(train_multi(train_loader, model, device, optimizer, epoch))
        val_loss.append(test_multi(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))
        if save_every_epoch or epoch == n_epochs-1:
            model_filename = os.path.join(path_to_save, f"{model_name}_e{epoch+1:03d}_{val_loss[epoch]:.5f}.pt")
            torch.save(model.state_dict(), model_filename)

    metrics = zip(train_loss, val_loss)
    with open(metrics_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Train_Loss", "Val_Loss"])
        writer.writerows(metrics)


if __name__ == "__main__":
    main()
