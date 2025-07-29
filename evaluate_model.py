import os 
import json
import time
import argparse 
from pprint import pprint


import tqdm
import torch 
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from sklearn import metrics 

from data import PrepAudioDataset, handle_bad_samples_collate_fn
from models import SSDNet1D
from helpers import plot_confusion_matrix


def evaluate(model_checkpoint_path: str,  dataset_loader: DataLoader):
    net = SSDNet1D()
    checkpoint = torch.load(model_checkpoint_path) 
    net.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")
    net.to(device)
    net.eval() 

    acc_accumulated = 0
    numsamples = 0
    inferences = []
    ground_truth = []


    with torch.no_grad():
        for batch in tqdm.tqdm(dataset_loader):
            if batch:
                samples, labels = batch 

                numsamples += len(labels)

                samples = samples.to(device)
                labels = labels.to(device)

                infer = net(samples)

                inferences.extend(infer.cpu().numpy().squeeze().tolist())
                ground_truth.extend(labels.cpu().numpy().squeeze().tolist())

                infer = infer.argmax(dim=1)
                batch_acc = infer.eq(labels).sum().item()
                acc_accumulated += batch_acc
            else:
                print("Skipping a batch due to missing all files in the batch")
        accuracy = round(acc_accumulated / numsamples, 2) 
        print(f"Accuracy: {accuracy}")

    return inferences, ground_truth


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--labels_filepath", type=str, required=True, help="labels space delimited file")
    parser.add_argument("--label_column_name", type=str, required=True, help="Name of the label column in the label file")
    parser.add_argument("--filename_column_name", type=str, required=True, help="Name of the filename column in the label file")
    parser.add_argument("--checkpoints_path", help="The path to the model checkpoint 'pth' file containing the state dictionary and other metadata")
    parser.add_argument("--output_dir", default="./results", help="Where to save the evaluation output")
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
    checkpoints_path = args.checkpoints_path
    output_dir = args.output_dir
    real_label = args.real_label 
    fake_label = args.fake_label 
    batch_size=args.batch_size

    os.makedirs(output_dir, exist_ok=True)

    dataset = PrepAudioDataset(source_directory, labels_filepath, label_column_name, filename_column_name, real_label, fake_label)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=handle_bad_samples_collate_fn)

    inferences, ground_truth = evaluate(checkpoints_path, loader)
    ground_truth = list(ground_truth)
    y_pred = [int(torch.tensor(m).argmax()) for m in inferences]
    y_score = [F.softmax(torch.tensor(m)) for m in inferences]
    score_fake = [m[1] for m in y_score]

    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, y_pred).ravel()
    accuracy = metrics.accuracy_score(ground_truth, y_pred)
    precision = metrics.precision_score(ground_truth, y_pred)
    recall = metrics.recall_score(ground_truth, y_pred)
    f1_score = metrics.f1_score(ground_truth, y_pred)
    tn=int(tn)
    fp=int(fp)
    fn=int(fn)
    tp=int(tp)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    roc_auc_score = metrics.roc_auc_score(ground_truth, score_fake)
    logloss = metrics.log_loss(ground_truth, score_fake)

    metrics_dict = dict(
        accuracy = accuracy,
        precision = precision,
        recall =recall,
        f1_score = f1_score,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        fpr = fpr,
        fnr = fnr,
        roc_auc_score = roc_auc_score,
        logloss = logloss
    )

    print("Evaluation results:")
    pprint(metrics_dict)

    time_name = time.ctime()
    time_name = time_name.replace(' ', '_')
    time_name = time_name.replace(':', '_')
    output_prefix = os.path.join(output_dir, time_name)
    os.makedirs(output_prefix, exist_ok=True)

    # save the inferences and ground truth 
    inferences_ground_truth = {
        "ground_truth": ground_truth,
        "inferences": inferences
    }
    with open(os.path.join(output_prefix, "evaluation_ground_truth_and_inferences.json"), "w") as f: 
        f.write(json.dumps(inferences_ground_truth))

    
    with open(os.path.join(output_prefix,"metrics.json"), "w") as f: 
        f.write(json.dumps(metrics_dict))

    print(f"metrics saved to {output_prefix}")

    plot_confusion_matrix(tp, fp, fn, tn, ["real", "fake"], "Confusion Matrix", os.path.join(output_prefix, "confusion_matrix.png"))
    print("Evaluation complete")
