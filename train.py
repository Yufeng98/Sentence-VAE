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
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid']
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = Data(split=split)
    model = SentenceVAE(
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

    # def kl_anneal_function(anneal_function, step, k, x0):
    #     if anneal_function == 'logistic':
    #         return float(1 / (1 + np.exp(-k * (step - x0))))
    #     elif anneal_function == 'linear':
    #         return min(1, step / x0)

    # NLL = torch.nn.NLLLoss(size_average=False)
    MSE = torch.nn.MSELoss()
    Cos = torch.nn.CosineSimilarity(dim=-1)

    def loss_fn(output, target, length, mean, logv, anneal_function, step, k, x0):
        # Negative Log Likelihood
        # NLL_loss = NLL(logp, target)
        mse = MSE(output, target)
        COS = 1 - Cos(output, target)
        cos = torch.mean(COS)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        # KL_weight = kl_anneal_function(anneal_function, step, k, x0)

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
                length = [9 for _ in range(20)]
                if torch.is_tensor(batch):
                    batch = to_var(batch)
                target = batch.clone()
                # Forward pass
                output, mean, logv, z = model(batch, length)

                # loss calculation
                cos, mse, KL_loss= loss_fn(output, target,
                                                       length, mean, logv, args.anneal_function, step, args.k,
                                                       args.x0)
                # print(cos.item(), mse.item(), KL_loss.item())
                loss = (cos + mse + KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
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

            # save a dump of all sentences and the encoded latent space
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

            # print(target, output)
            # print(torch.abs((target-output)/target))
            if (epoch == args.epochs) and (split == "valid"):
                x = np.arange(32)
                save = {}
                save['target'] = target.cpu().detach().numpy().tolist()
                save['output'] = output.cpu().detach().numpy().tolist()
                with io.open('./' + '/save.json', 'wb') as data_file:
                    data = json.dumps(save, ensure_ascii=False)
                    data_file.write(data.encode('utf8', 'replace'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=8)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
