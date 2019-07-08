import os
import sys
import json
import numpy
import logging
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from models import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
torch.set_num_threads(4)


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class ToyModel(object):
    def __init__(self, task_config):
        self.model_config = load_json(task_config['model_config_file'])
        self.model_size = self.model_config['size']
        self.factor = self.model_config['factor']
        self.parameters = [
            numpy.ones((self.model_size, self.model_size)) * 0
        ]

    def get_weights(self):
        return self.parameters

    def set_weights(self, parameters):
        self.parameters = parameters

    def train_one_epoch(self):
        "return training loss and accuracy"
        self.parameters = [
            param + self.factor for param in self.parameters
        ]
        return 1, 2

    def validate(self):
        return self.parameters[0][0, 0], self.parameters[0][0, 0]

    def evaluate(self):
        return self.parameters[0][0, 0], self.parameters[0][0, 0]


class Yolo(object):
    def __init__(self, task_config):
        self.task_config = task_config
        self.model_config = load_json(task_config['model_config'])
        self.log_filename = task_config['log_filename']
        print(self.model_config)
        self.dataset = ListDataset(self.task_config['train'],
                                   augment=True,
                                   multiscale=self.model_config['multiscale_training'])
        logging.info('load data')
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.task_config['batch_size'],
                                     shuffle=True,
                                     num_workers=self.task_config['n_cpu'],
                                     collate_fn=self.dataset.collate_fn)
        # TODO: add a valset for validate
        self.testset = ListDataset(self.task_config['test'],
                                   augment=False,
                                   multiscale=False)
        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.task_config['batch_size'],
            num_workers=1,
            shuffle=False,
            collate_fn=self.testset.collate_fn
        )
        self.train_size = self.dataset.__len__()
        print("train_size:", self.train_size)
        self.valid_size = self.testset.__len__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = Darknet(self.model_config['model_def']).to(self.device)
        logging.info('model construct completed')
        self.best_map = 0
        self.optimizer = torch.optim.Adam(self.yolo.parameters())


    def get_weights(self):
        params = [param.data.cpu().numpy()
                  for param in self.yolo.parameters()]
        return params

    def set_weights(self, parameters):
        for i, param in enumerate(self.yolo.parameters()):
            param_ = torch.from_numpy(parameters[i]).cuda()
            param.data.copy_(param_)

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        self.yolo.train()
        for batch_i, (_, imgs, targets) in enumerate(self.dataloader):
            batches_done = len(self.dataloader) * 1 + batch_i
            imgs = Variable(imgs.to(self.device))
            targets = Variable(targets.to(self.device), requires_grad=False)
            loss, outputs = self.yolo(imgs, targets)
            loss.backward()
            if batch_i % 10 == 0:
                print("step: {} | loss: {:.4f}".format(batch_i, loss.item()))
            if batches_done % self.model_config["gradient_accumulations"]:
                # Accumulates gradient before each step
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss.item()

    def eval(self, dataloader, yolo, test_num=10000):
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        total_losses = list()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets = Variable(targets.to(self.device), requires_grad=False)

            imgs = Variable(imgs.type(Tensor), requires_grad=False)
            with torch.no_grad():
                loss, outputs = yolo(imgs, targets)
                outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5)
                total_losses.append(loss.item())
            targets = targets.to("cpu")
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= int(self.model_config['img_size'])
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
        if len(sample_metrics) > 0:
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        else:
            precision, recall, AP, f1, ap_class = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
        total_loss = sum(total_losses) / len(total_losses)
        return total_loss, AP.mean(), recall.mean()

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set, 
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, mAP, recall = self.eval(
            self.test_dataloader, self.yolo, test_num)
        return total_loss, mAP, recall


class Models:
    ToyModel = ToyModel
    Yolo = Yolo


if __name__ == "__main__":
    # unit tests for each wrapped model
    pass
