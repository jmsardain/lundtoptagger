# lundtoptagger

Tag top and W jets using the LundNet model.


## Setup

On UChicago, samples and flat weights are here:  
`/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/`

The included setup script can set up the environment on several different systems:

- a system with Red Hat Enterprise Linux 9, an NVIDIA driver which supports CUDA >= 11.8, and access to CVMFS, such as `lxplus-gpu`
- a system with CentOS 7 and access to CVMFS (currently set up without CUDA)
- UCL's `gpu02` server and Hypatia GPU partition

The script will automatically figure out which of these systems it is running on and set up the environment accordingly; just do

```bash
source setup.sh
```

On UChicago, do

```bash
source /data/jmsardain/LJPTagger/JetTagging/miniconda/bin/activate
conda activate rootenv
```

## Data preparation

To create graphs for training from ROOT files and save them to a file, first process JETM2 or FTAG1 derivations with the following code:  
<https://gitlab.cern.ch/rvinasco/jetmdatamc/-/tree/temporaryRun2>  
Then run `Make_data.py` on the output:

```bash
python Make_data.py configs/config_make_data.yaml
```

The script applies selections defined in the configuration files, creates Lund trees (graphs) for each jet, and calculates weights which make the jet $p_T$ distribution flat.
Two files are created:
a file containing a list of graphs (`torch_geometric.data.Data` objects) that can be used for training and testing the tagging model,
and a ROOT file containing some properties of the jets passing selection:

- DSID (MC channel number) of the dataset from the which the jet was taken
- MC event weight
- mass, $p_T$, $\eta$, $\phi$, and number of charged constituents of the jets
- weight which makes the $p_T$ distribution flat
- large-R jet truth labels (1 for top, 2 for W, 10 for QCD)
- signal/background label (1 for signal, 0 for background)
- GN2X scores (if available)

Some of these are already stored as attributes of the graphs, and they could all be, but the graphs can take a long time to load,
so it can be useful to have a separate file for plots which don't require the Lund trees.

### Configuration and parameters for `Make_data.py`

The configuration is defined in `configs/config_make_data.yaml`. The path to this config file must be given as a command-line argument, as in the example above.

In this file, you can set:

- the input and output file paths,
- fractions of the data to save in separate files - this can be used for train/test splits, or memory management,
as the events are loaded and processed in chunks of sizes determined by these fractions
- a value for the optional $k_T$ cut

The script uses another configuration file, `config_signal.yaml`, which contains parameter sets for several signal samples.
The path to this file and the choice of parameter set from it are also specified in `config_make_data.yaml` under the `signal_config_file` and `signal` keys, respectively.
The parameters in `config_signal.yaml` include values for the selection cuts (mass, $p_T$, minimum number of splittings)
and paths to files with histograms of the $p_T$ distributions of the jets, which are used to calculate the $p_T$ weights
so that they are proportional to 1/(bin count).
These histograms are included in the repository; they are located in the `histos` folder.
They can be created with the `make_histos.py` script, which also applies mass and $pT$ cuts from `config_signal.yaml`.

Rather than choosing a single signal configuraion, the `signal` key in `config_make_data.yaml` can also be set to `all`,
in which case the script will combine the selections (to include jets which pass any of the selections)
and store multiple sets of flat-pT weights (in both the graphs and ROOT files), one for each signal configuration.
The main reason for this is that you don't have to process the background jets (QCD) multiple times,
and there is no need to have multiple QCD graphs files with slightly different selections.
Instead, when later using the file, you can choose which set of weights to use and apply corresponding selection cuts.
If saving multple sets of weights, they will be saved with the signal configuration identifier as a suffix in the branch/attribure names.
You can also do this with a single signal configuration by setting `signal_name_in_weight` to `True` in `config_make_data.yaml`,
in order to have matching names between the signal files where you would probably only use 1 configuration and the background files where you might want to use multiple configurations.

You can override any of the parameters in `config_make_data.yaml` except `event_fractions` using the `--override` command-line argument; for example:

```bash
python Make_data.py configs/config_make_data.yaml --override path_to_rootfiles="/path/to/root/files/*.root" id="QCD" kT_cut=0.5
```

The values for the override arguments should be in the YAML format - e.g. `null` will be interpreted as `None` and `.inf` as `float('inf')`.

Warning: if you override `event_fractions`, the code will run without error, but it will only be done correctly if you use the all the same keys. So it's best to avoid overriding it via the command line and instead edit the config file directly.


## Training

For the training, the main changes one should do are in the configuration file: `config_ONLY_TRAIN.yaml`.
In this file you will define the learning rate, batch size, the input files, the model to use, the location to save your checkpoints.

To run the training:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml
```

There are two optional arguments which can be used to override the values in the config file:

- `--ln_kT_cut`: float
- `--do_combined_training`: value can be true/false, yes/no, 0/1, case insensitive

For example:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml --ln_kT_cut 0 --do_combined_training true
```

## Testing

Run the testing:

```bash
python test_make_scores.py configs/config_make_scores.yaml
```

Some paths and names in the configuration file can have placeholders that are replaced by values of other parameters,
namely by the values of `kT_cut` and `sample`.
This makes it easy to run on different samples:
if you keep the paths to your samples and output files the same apart from a single segment that is different for every sample,
you can just change the `sample` parameter in the config file without having to change several different variables
(`paths_to_test_file_root`, `paths_to_test_file_graphs`, `path_to_outdir`, and `output_name`).

In the configuration file, you can also specify the name of the branch to save the scores to.
If the specified output already exists and has a branch with the same name,
the branch will be overwritten; otherwise, it will be added to the file.
In both cases the other branches will be preserved.
This means you can run multiple times on the same test files and with the same output file,
changing the model and score branch name each time to produce a single ROOT file with multiple sets of scores.

You can override any of the parameters in `config_make_scores.yaml` using the `--override` command-line argument. To override a value specified in the config file like this:

```yaml
key:
  subkey: value
```

you can use the following syntax:

```bash
python test_make_scores.py configs/config_make_scores.yaml --override key.subkey=value
```

For example:

```bash
python test_make_scores.py configs/config_make_scores.yaml --override data.sample=Sherpa_Cluster data.kT_cut=0
```


## Plotting

To plot using the code in the plotting folder, you must first combine the ROOT file outputs of the testing scripts
into a single file:

```bash
hadd -f tree.root user.*root
```

In a clean and new terminal, go to the plotting repo and source the setup file. 
It will get the version of the libraries you want to use from /cvmfs/. 
Go to plotting.py and check that you are using the root file you just created with hadd after the testing of the model. 
Plot! 
```
source setup.sh
python -b plotting.py 
```

## To do list: 
- [ ] Cut on ln(kt): prepare multiple graphs with different values of ln(kT) cuts 
- [ ] Make a bkg rej vs ln(kT) plot
- [ ] Make the LundJetPlane plot with the prediction to see where the modeling uncertainties impact the most
- [ ] Apply a shift of 5% to mean pT of the constituent, and test on that sample
- [ ] Apply a shift of 5% to resolution pT of the constituent, and test on that sample

# LundNet MultiClass
This section is to describe the multi-class workflow. Currently, `LundNet4Class` is a copy of `LundNet`, but with four output nodes. `log_softmax()` activation has been used as the final activation function. Many dedicated utility functions are written in `utils_multiclass.py`
## Data Preparation

Uses the same `Make_data.py` script. Graph making function `create_train_dataset_fulld_new_Ntrk_pt_weight_file` has been modified to return custom jet labels. See `configs_FourProng/config_make_data.yaml` as an example. User can specify the label branch name other than the truth label.
```shell
python3 Make_data.py configs_FourProng/config_make_data.yaml
```
### Data Shuffling

For the multi-class training, we expect large datasets. Some computing clusters may not have job nodes with large memory capacity, thus lazy dataset loading has been implemented. See `Iterative Dataset` below. It uses `pandas.df` to map jets between the root files and the graph files.

- Jet weighting: It first loads all the root files to fill the pt, mass, and pt-mass histograms. Then a flattening is applied creating weight per bin
- Class balancing: Sum of all jet's weight per class may be imbalanced. Scale factor is applied to the other classes w.r.t. reference class
- Save dataset: Shuffled datasets are saved, see `save_split()` function.

Caveat: Weight calculation uses all jets in the root files, while the balancing uses only the selected number of jets.

During training, this means torch will all graph files per epoch, making the training time longer while minimizing the RAM usage(e.g. UChicagoAF)

```shell
python3 Shuffle_data.py configs_FourProng/config_shuffle_data.yaml
```
## Training
See `Train_MultiClass.py`. It's a training macro based on `weight_ONLY_TRAINS.py`

```shell
python3 Train_MultiClass.py configs_FourProng/config_Train_MultiClass.yaml
```

## Testing
`Make_Score_MultiClass.py` writes the `LundNet4Class` model score to the .root files created from `Make_Data.py`. It can loop through multiple LundNet checkpoints under `path_to_combined_ckpt`. Now a user can use the score distribution to assess the `LundNet4Class` performance.
```shell
python3 Make_Score_MultiClass.py configs_FourProng/config_make_scores_MultiClass.yaml
```