# Sentence Variational Autoencoder

Put `UM`, `NYU`, `USM`, `UCLA` ABIDE I Preprocessing Dataset in current directory.
Run `sh get_latent_for_sites`. `${SITE}_data.json` is shuffled and split dataset. `${SITE}_latent.json` stores produced latent features. `${SITE}_save.json` stores input and output of VAE in the last validation set, we can use which to plot and compare input and output.

## Data Precrocessing

|Site           |UM     |NYU    |USM    |UCLA   |
|:-------------:|:-----:|:-----:|:-----:|:-----:|
|Subject        |95     |173    |70     |71     |
|Lines          |288    |175    |232    |112    |
|Seq_len        |9      |7      |8      |7      |
|Embedding_size |32     |25     |29     |16     |

For example, in site `UM`, shape of dataset loaded from raw file is `[95, 200, 9, 32]`, where `95` is the total number of subjects, `200` is number of regions of interests (ROIs) or nodes in graph, `9` is length of sequence in LSTM and `32` is embedding size in LSTM and also length of truncated sequence used in correlation matrix. 76 out of 95 subjects (80%) are used as training dataset, while other 19 subjects (20%) are used as validation dataset. Dataset has been shuffled before splitting. There are `760` batches in training set and `190` batches in validation set in `UM`. Regulations of data preprocessing in `NYU`, `USM`, `UCLA` are similar.

## Model

Structure of LSTM_VAE model is shown as follow.

LATM_VAE(
    (embedding_dropout): Dropout(p=0.5)
    (encoder_rnn): GRU(embedding_size, hidden_size=256, batch_first=True)
    (hidden2mean): Linear(in_features=256, out_features=8, bias=True)
    (hidden2logvar): Linear(in_features=256, out_features=8, bias=True)
    (latent2hidden): Linear(in_features=8, out_features=256, bias=True)
    (decoder_rnn): GRU(embedding_size, hidden_size=256, batch_first=True)
    (hidden2embedding): Linear(in_features=256, out_features=16, bias=True)
)

Loss function is combination of MSE Loss, Cosine Similarity and KL-Divergence.