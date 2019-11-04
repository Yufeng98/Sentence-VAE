# Sentence Variational Autoencoder

Put `UM`, `NYU`, `USM`, `UCLA` ABIDE I Preprocessing Dataset in current dir.
Run `sh get_latent_for_sites`.

## Data Precrocessing

|Site           |UM     |NYU    |USM    |UCLA   |
|:-------------:|:-----:|:-----:|:-----:|:-----:|
|Subject        |95     |173    |70     |71     |
|Lines          |288    |175    |232    |112    |
|Seq_len        |9      |7      |8      |7      |
|Embedding_size |32     |25     |29     |16     |

For example, in site `UM`, shape of dataset loaded from raw file is `[95, 200, 9, 32]`, where `95` is the total number of subjects, `200` is number of regions of interests (ROIs) or node in graph, `9` is length of sequence in LSTM and `32` is embedding size in LSTM and also length of truncated sequence used in correlation matrix. 76 out of 95 subjects (80%) are used as training dataset, while other 19 subjects (20%) are used as validation dataset. There are 15200 items (`[9, 32]`) in training dataset in `760` batches, where batch size is `20` and 380 items (`[9, 32]`) in validation dataset in `190` batches, where batch size is `20`.

## Model
