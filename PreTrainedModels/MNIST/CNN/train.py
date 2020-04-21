from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import os

from .. import utils
from DeepDIGCode.PreTrainedModels.MNIST.CNN.model import Model
from DeepDIGCode import config

args = config.args

def main():

    save_dir = args.project_dir+'PreTrainedModels/'+args.dataset+'/'+args.pre_trained_model+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_enabled = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_enabled else "cpu")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Model().to(device)
    if os.path.exists(save_dir+'pre_trained_model.m'):
        model =torch.load(save_dir+'pre_trained_model.m')
        print("Pre-trained model loaded")

    train_data, train_labels = utils.get_original_data(split='train')
    test_data, test_labels = utils.get_original_data(split='test')
    def test(save_results = False):
        with torch.no_grad():
            predictions = []
            for i in range(0,len(test_data),100):
                start = i
                end = min(start+100, len(test_data))
                batch =test_data[start:end].to(device)
                output,_ = model(batch)
                pred = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
                predictions.append(pred)

        predictions = np.concatenate(predictions)
        ground_truths = test_labels.cpu().numpy()
        accuracy = accuracy_score(y_true=ground_truths,y_pred=predictions)
        f1_weighted = f1_score(y_true=ground_truths,y_pred=predictions,average='weighted')
        if not save_results:
            print("Accuracy {} F1 {} ".format(accuracy,f1_weighted))
            return

        f1_micro = f1_score(y_true=ground_truths,y_pred=predictions,average='macro')
        f1_macro = f1_score(y_true=ground_truths,y_pred=predictions,average='micro')

        report = classification_report(y_true=ground_truths,y_pred=predictions)
        conf_matrix = confusion_matrix(y_true=ground_truths,y_pred=predictions)

        performance_file = open(save_dir+'test_performance_log.txt','w')
        performance_file.write("Accuracy,{}\n".format(accuracy))
        performance_file.write("F1 weighted,{}\n".format(f1_weighted))
        performance_file.write("F1 macro,{}\n".format(f1_macro))
        performance_file.write("F1 micro,{}\n".format(f1_micro))
        performance_file.write("classification_report:\n{}\n".format(report))
        performance_file.write("conf_matrix:\n{}\n".format(conf_matrix))
    def train():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
        indices = [i for i in range(len(train_data))]
        for epoch in range(40):
            random.shuffle(indices)
            for i in range(0,len(train_data),64):
                start = i
                end = min(start+100, len(train_data))
                X,Y = train_data[indices[start:end]], train_labels[indices[start:end]]

                a,_= model(X.to(device))
                loss = criterion(a, Y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            torch.save(model,save_dir+'pre_trained_model.m')
            print(epoch, loss.data)
            test(save_results=False)

    train()
    test(save_results=True)
if __name__ == '__main__':
    main()
