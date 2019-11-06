import os
import io
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from data import Data
from utils import to_var, expierment_name
from model import LSTM_VAE


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    # Load dataset
    splits = ['train', 'valid']
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = Data(split, args.batch_size, args.site, args.subject, args.seq_len,
                               args.embedding_size, args.cut_start, args.lines)

    # load model
    model = LSTM_VAE(
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,  # gru
        hidden_size=args.hidden_size,  # 256
        word_dropout=args.word_dropout,  # 0
        embedding_dropout=args.embedding_dropout,  # 0.5
        latent_size=args.latent_size,  # 8
        num_layers=args.num_layers,  # 1
        bidirectional=args.bidirectional  # false
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)
    """
    SentenceVAE(
      (embedding_dropout): Dropout(p=0.5)
      (encoder_rnn): GRU(32, 256, batch_first=True)
      (decoder_rnn): GRU(32, 256, batch_first=True)
      (hidden2mean): Linear(in_features=256, out_features=8, bias=True)
      (hidden2logv): Linear(in_features=256, out_features=8, bias=True)
      (latent2hidden): Linear(in_features=16, out_features=256, bias=True)
    )
    """
    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    # NLL = torch.nn.NLLLoss(size_average=False)
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity(dim=-1)

    def loss_fn(output, target, mean, logvar):
        mse = mse_loss(output, target)
        cos = torch.mean(1 - cos_loss(output, target))
        KL_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return cos, mse, KL_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    for epoch in range(1, args.epochs + 1):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = args.batch_size
                batch = batch.type(torch.float32)
                length = [args.seq_len for _ in range(20)]
                if torch.is_tensor(batch):
                    batch = to_var(batch)
                target = batch.clone()

                # Forward pass
                output, mean, logvar, z = model(batch, length)

                # loss calculation
                cos, mse, KL_loss = loss_fn(output, target, mean, logvar)
                # print(cos.item(), mse.item(), KL_loss.item())
                loss = (cos + mse + KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # book keeping
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.detach().reshape(1)))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/Cos Loss" % split.upper(), cos.item() / batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/MSE Loss" % split.upper(), mse.item() / batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch * len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration + 1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, Cos-Loss %9.4f, MSE-Loss %9.4f, KL-Loss %9.4f"
                          % (split.upper(), iteration, len(data_loader) - 1, loss.item(), cos.item() / batch_size,
                             mse.item() / batch_size, KL_loss.item() / batch_size))

                # if split == 'valid':
                #     if 'target_sents' not in tracker:
                #         tracker['target_sents'] = list()
                #     tracker['target_sents'] += idx2word(batch['target'].detach(), i2w=datasets['train'].get_i2w(),
                #                                         pad_idx=datasets['train'].pad_idx)
                #     tracker['z'] = torch.cat((tracker['z'], z.detach()), dim=0)

            print(
                "%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # # save a dump of all sentences and the encoded latent space
            # if split == 'valid':
            #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
            #     if not os.path.exists(os.path.join('dumps', ts)):
            #         os.makedirs('dumps/' + ts)
            #     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
            #         json.dump(dump, dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)

            # save target & output for last validation batch
            if (epoch == args.epochs) and (split == "valid"):
                save = {'target': target.cpu().detach().numpy().tolist(),
                        'output': output.cpu().detach().numpy().tolist()}
                with io.open('./{}_save.json'.format(args.site), 'wb') as data_file:
                    data = json.dumps(save, ensure_ascii=False)
                    data_file.write(data.encode('utf8', 'replace'))

    # save latent space
    latent = []
    for split in splits:

        data_loader = DataLoader(
            dataset=datasets[split],
            batch_size=args.batch_size,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        model.eval()

        for iteration, batch in enumerate(data_loader):

            batch = batch.type(torch.float32)
            length = [args.seq_len for _ in range(20)]
            if torch.is_tensor(batch):
                batch = to_var(batch)

            # Forward pass
            output, mean, logv, z = model(batch, length)

            # save latent space for both training and validation batch
            latent.append(z.cpu().detach().numpy().tolist())
    latent = np.array(latent).reshape(args.subject, 200, args.latent_size)
    print(np.shape(latent))
    with io.open('./{}_latent.json'.format(args.site), 'wb') as data_file:
        data = json.dumps(latent.tolist(), ensure_ascii=False)
        data_file.write(data.encode('utf8', 'replace'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--site', type=str, default='UM')
    parser.add_argument('--cut_start', type=int, default=4)
    parser.add_argument('--lines', type=int, default=288)
    parser.add_argument('--subject', type=int, default=95)
    parser.add_argument('--seq_len', type=int, default=9)
    parser.add_argument('-eb', '--embedding_size', type=int, default=32)

    # do not need to change
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=8)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    # not used parameters
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
