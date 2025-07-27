import argparse 

import torch 
from torch.utils.data.dataloader import DataLoader

import tqdm
from data import PrepAudioDataset, handle_bad_samples_collate_fn
from models import SSDNet1D


def evaluate(model_checkpoint_path: str,  dataset_loader: DataLoader):
    net = SSDNet1D()
    checkpoint = torch.load(model_checkpoint_path) 
    net.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")
    net.to(device)
    net.eval() 

    softmax_acc = 0
    numsamples = 0
    inferences = []
    ground_truth = []


    with torch.no_grad():
        for batch in tqdm.tqdm(dataset_loader):
            samples, labels = batch 

            numsamples += len(labels)

            samples.to(device)
            labels.to(device)

            infer = net(samples)

            inferences.extend(list(infer.numpy()))
            ground_truth.extend(list(labels))

            
            t1 = F.softmax(infer, dim=1)
            t2 = labels.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc
        accuracy = round(softmax_acc / numsamples, 2) 
        print(f"Accuracy: {accuracy}")

    return inferences, ground_truth


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--labels_filepath", type=str, required=True, help="labels space delimited file")
    parser.add_argument("--label_column_name", type=str, required=True, help="Name of the label column in the label file")
    parser.add_argument("--filename_column_name", type=str, required=True, help="Name of the filename column in the label file")
    parser.add_argument("--checkpoints_dir", default="./models", help="Where to save model checkpoints or load checkpoints from")
    parser.add_argument("--real_label", default='real', help="Name of the 'real' label")
    parser.add_argument("--fake_label", default='fake', help="Name of the 'fake' label")
    parser.add_argument("--batch_size", default=64, help="Batch size used for loading data")
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    source_directory = args.source_directory 
    labels_filepath = args.labels_filepath
    label_column_name = args.label_column_name 
    filename_column_name = args.filename_column_name 
    checkpoints_dir = args.checkpoints_dir
    real_label = args.real_label 
    fake_label = args.fake_label 
    batch_size=args.batch_size



    dataset = PrepAudioDataset(source_directory, labels_filepath, label_column_name, filename_column_name, real_label, fake_label)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=handle_bad_samples_collate_fn)

    inferences, ground_truth = evaluate(checkpoints_dir, loader)

    