# Sentence Variational Autoencoder

## Data Precrocessing

`data.py` contains the class Data, dataset will be loaded from which when training.

Shape of dataset loaded from raw file is `[95, 200, 9, 32]`, where `95` is the total number of subjects, `200` is number of regions of interests (ROIs) or node in graph, `9` is length of sequence in LSTM and `32` is embedding size in LSTM and also length of truncated sequence used in correlation matrix.

80 out of 95 subjects are used as training dataset, while other 15 subjects are used as validation dataset. There are 16000 items (`[9, 32]`) in training dataset and `800` batches, where batch size is `20`.

## Model
