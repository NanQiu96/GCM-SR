# GCM-SR



## Paper data and code

This is the source code for the Paper: Incorporating Global Context into Multi-task Learning for Session-based Recommendation.

##### Datasets

You can download the three datasets (Diginetica, Tmall and Nowplaying) that we used in the paper from https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1. After downloaded the datasets, please put them in the folder 'datasets/'.



## Requirements

- Python 3
- PyTorch 1.5.0
- tqdm



## Usage

You need to run the data processing file first to preprocess the corresponding data.

For example: 

```
python preprocess.py --dataset diginetica
python process_tmall.py --dataset Tmall
python process_nowplaying.py --dataset Nowplaying
```



Next, you need to calculate the neighbors of each item in all sessions.

For example:

```
python build_graph.py --dataset diginetica --sample_num 12

usage: build_graph.py [-h] [--dataset DATASET] [--sample_num SAMPLE_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     diginetica/Tmall/Nowplaying
  --sample_num SAMPLE_NUM
```



Finally, you can run the file 'main.py' to train and evaluate the model:

For example: 

```
# Diginetica
python main.py --dataset diginetica --hop 2 --omega 0.2
# Tmall
python main.py --dataset Tmall --hop 2 --omega 0.2
# Nowplaying
python main.py --dataset diginetica --hop 1 --omega 0.6

usage: main.py [-h] [--dataset DATASET] [--hiddenSize HIDDENSIZE]
               [--epoch EPOCH] [--n_sample_all N_SAMPLE_ALL]
               [--n_sample N_SAMPLE] [--batch_size BATCH_SIZE] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2] [--hop HOP]
               [--validation] [--valid_portion VALID_PORTION] [--alpha ALPHA]
               [--patience PATIENCE] [--norm NORM] [--scale SCALE] [--tau TAU]
               [--omega OMEGA]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     diginetica/Nowplaying/Tmall
  --hiddenSize HIDDENSIZE
  --epoch EPOCH
  --n_sample_all N_SAMPLE_ALL
  --n_sample N_SAMPLE
  --batch_size BATCH_SIZE
  --lr LR               learning rate.
  --lr_dc LR_DC         learning rate decay.
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate decay.
  --l2 L2               l2 penalty
  --hop HOP
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion
  --alpha ALPHA         Alpha for the leaky_relu.
  --patience PATIENCE
  --norm NORM           adapt NISER, l2 norm over item and session embedding
  --scale SCALE         scaling factor sigma
  --tau TAU             scale factor of the scores.
  --omega OMEGA         weight of the global task.
```

