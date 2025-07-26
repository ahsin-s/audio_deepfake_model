import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from data import PrepAudioDataset
import models
from test import asv_cal_accuracies, cal_roc_eer
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import argparse 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", required=True, type=str)
    parser.add_argument("--labels_filepath", required=True, help="labels space delimited file")
    parser.add_argument("--label_column_name", required=True, help="Name of the label column in the label file")
    parser.add_argument("--filename_column_name", required=True, help="Name of the filename column in the label file")
    parser.add_argument("--checkpoints_dir", default="./models", help="Where to save model checkpoints or load checkpoints from")
    parser.add_argument("--real_label", default='real', help="Name of the 'real' label")
    parser.add_argument("--fake_label", default='fake', help="Name of the 'fake' label")
    parser.add_argument("--batch_size", default=64, help="Batch size used for loading data")
    return parser



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    args = parser.parse_args()

    source_directory = args.source_directory 
    labels_filepath = args.labels_filepath,
    label_column_name = args.label_column_name 
    filename_column_name = args.filename_column_name 
    checkpoints_dir = args.checkpoints_dir
    real_label = args.real_label 
    fake_label = args.fake_label 
    batch_size=args.batch_size


    os.makedirs(checkpoints_dir, exist_ok=True)


    train_set = PrepAudioDataset(source_directory, labels_filepath, label_column_name, filename_column_name, real_label, fake_label)
    weights = train_set.get_weights().to(device)  # weight used for WCE
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    Net = models.SSDNet1D()   # Res-TSSDNet
    Net = Net.to(device)

    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = 'WCE'  # {'WCE', 'mixup'}

    print('Training data: {}, Date type: {}.'.format(train_data_path, data_type))

    num_epoch = 100
    loss_per_epoch = torch.zeros(num_epoch,)
    best_d_eer = [.09, 0]

    log_path = f'{checkpoints_dir.rstrip("/")}/train_log/'
    os.makedirs(log_path, exist_ok=True)
    print(f"Writing logs to {log_path}")

    time_name = time.ctime()
    time_name = time_name.replace(' ', '_')
    time_name = time_name.replace(':', '_')
    f = open(log_path + time_name + '.csv', 'w+')

    print("Starting training")
    for epoch in range(num_epoch):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        print(f"Epoch {epoch} of {num_epoch}")
        for batch in tqdm.tqdm(train_loader):
            counter += 1
            # forward
            samples, labels, _ = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if loss_type == 'mixup':
                # mixup
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(labels))
                samples = lam*samples + (1-lam)*samples[index, :]
                preds = Net(samples)
                labels_b = labels[index]
                loss = lam * F.cross_entropy(preds, labels) + (1 - lam) * F.cross_entropy(preds, labels_b)
            else:
                preds = Net(samples)
                loss = F.cross_entropy(preds, labels, weight=weights)
                # loss = F.cross_entropy(preds, labels)

            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss/counter

        dev_accuracy, d_probs = asv_cal_accuracies(dev_protocol_file_path, dev_data_path, Net, device, data_type=data_type, dataset=dataset)
        d_eer = cal_roc_eer(d_probs, show_plot=False)
        if d_eer <= best_d_eer[0]:
            best_d_eer[0] = d_eer
            best_d_eer[1] = int(epoch)

            eval_accuracy, e_probs = asv_cal_accuracies(eval_protocol_file_path, eval_data_path, Net, device, data_type=data_type, dataset=dataset)
            e_eer = cal_roc_eer(e_probs, show_plot=False)
        else:
            e_eer = .99
            eval_accuracy = 0.00

        net_str = data_type + '_' + str(epoch) + '_' + 'ASVspoof20' + str(dataset) + '_LA_Loss_' + str(round(total_loss / counter, 4)) + '_dEER_' \
                            + str(round(d_eer * 100, 2)) + '%_eEER_' + str(round(e_eer * 100, 2)) + '%.pth'
        torch.save({'epoch': epoch, 'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_per_epoch}, ('./trained_models/' + net_str))

        elapsed = time.time() - t

        print_str = 'Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, d_acc: {:.2f}%, e_acc: {:.2f}%, ' \
                    'dEER: {:.2f}%, eEER: {:.2f}%, best_dEER: {:.2f}% from epoch {}.'.\
                    format(epoch, elapsed/60, optimizer.param_groups[0]['lr']*1000, total_loss / counter, dev_accuracy * 100,
                           eval_accuracy * 100, d_eer * 100, e_eer * 100, best_d_eer[0] * 100, int(best_d_eer[1]))
        print(print_str)
        df = pd.DataFrame([print_str])
        df.to_csv(log_path + time_name + '.csv', sep=' ', mode='a', header=False, index=False)

        scheduler.step()

    f.close()
    plt.plot(torch.log10(loss_per_epoch))
    plt.show()

    print('End of Program.')
