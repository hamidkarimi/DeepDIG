import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import sys
import importlib
import pickle

from DeepDIGCode import config
from DeepDIGCode import utils

args = config.args

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model)
sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset)

dataset_utils = importlib.import_module("DeepDIGCode.PreTrainedModels." + args.dataset + ".utils")
from model import *
pre_trained_model =torch.load(args.project_dir+'PreTrainedModels/'+
                              args.dataset+'/'+ args.pre_trained_model+'/pre_trained_model.m').to(utils.device)

pre_trained_model.eval()

def train_s_t():

    save_dir = utils.class_s_t_dir +'adv/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    train_data, train_labels = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='train')
    def train():
        model = utils.ae_module.AE(dropout=args.dropout).to(utils.device)
        if os.path.exists(save_dir+'model.m'):
            model = torch.load(save_dir+'model.m')
            print("Saved model {} loaded".format(save_dir+'model.m'))
        else:
            print(" No trained model, creating one ...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=args.step_size_scheduler, gamma=args.gamma_scheduler)

        for step in range(args.steps):
            perm = torch.randperm(len(train_data))
            batch = train_data[perm[0:args.batch_size]].float().to(utils.device)
            Z,_ = pre_trained_model(batch.view(utils.get_pre_trained_model_input_shape()))
            pre_trained_model_data_probs = F.softmax(Z, dim=1)

            _,recon_batch = model(batch)

            Z,_= pre_trained_model(recon_batch.view(utils.get_pre_trained_model_input_shape()).to(utils.device))
            pre_trained_model_reconst_probs = F.softmax(Z, dim=1)
            loss = utils.loss_function_ae(recon_batch, batch, pre_trained_model_reconst_probs,
                                              pre_trained_model_data_probs, args.batch_size,
                                              utils.classes['t'], device=utils.device)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            scheduler.step()

            if step % 2000==0:
                print("Step {} Loss {}".format(step,loss.cpu().data))
        torch.save(model,save_dir+'model.m')
    def inference(split = 'test'):
        if os.path.exists(save_dir+'model.m'):
            model = torch.load(save_dir+'model.m')
            print("Saved model {} loaded".format(save_dir+'model.m'))
        else:
            print("NO trained model. Exit")
            exit(-1)
        if split =='test':
            data, labels = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='test')
        else:
            data, labels = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='train')

        with torch.no_grad():
            model.eval()
            targets = [utils.classes['s'] for _ in range(len(data))]
            ground_truth = labels.cpu().numpy()
            recon_batches, pre_trained_model_reconst_predictions,pre_trained_model_data_predictions = [],[],[]
            recon_probs = []
            for i in range(0,len(data),args.batch_size):
                start = i
                end = min(start+args.batch_size,len(data))
                recon_batch = model(data[start:end].to(utils.device))[1]

                recon_batches.append(recon_batch)
                Z,_ = pre_trained_model(data[start:end].
                                              view(utils.get_pre_trained_model_input_shape(end - start)).to(utils.device))

                pre_trained_model_data_predictions.append(np.argmax(F.softmax(Z, dim=1).cpu().numpy(),axis=1))

                Z,_ = pre_trained_model(recon_batch.
                                              view(utils.get_pre_trained_model_input_shape(end - start)).to(utils.device))

                recon_probs.append(F.softmax(Z, dim=1).cpu().numpy())
                pre_trained_model_reconst_predictions.append(np.argmax(F.softmax(Z, dim=1).cpu().numpy(),axis=1))
            recon_probs = np.concatenate(recon_probs)
            recon_batches = torch.cat(recon_batches)
            pre_trained_model_reconst_predictions = np.concatenate(pre_trained_model_reconst_predictions)
            pre_trained_model_data_predictions = np.concatenate(pre_trained_model_data_predictions)


            accuracy_score_reconst_target = accuracy_score(y_true=targets,y_pred=pre_trained_model_reconst_predictions)
            f1_score_reconst_target = f1_score(y_true=targets,y_pred=pre_trained_model_reconst_predictions,average='weighted')

            accuracy_score_data_groundtruth = accuracy_score(y_true=ground_truth,y_pred=pre_trained_model_data_predictions)
            f1_score_data_groundtruth = f1_score(y_true=ground_truth,y_pred=pre_trained_model_data_predictions,average='weighted')

            confmatix = confusion_matrix(y_true=targets,y_pred=pre_trained_model_reconst_predictions)

            with open(save_dir+'{}_adv_{}_{}.txt'.format(split,utils.classes['s'],utils.classes['t']),'w') as fp:
                    fp.write("Accurcay Reconst Target, {}\n".format(accuracy_score_reconst_target))
                    fp.write("F1-score Reconst Target, {}\n".format(f1_score_reconst_target))
                    fp.write("Accurcay Data GroundTruth, {}\n".format(accuracy_score_data_groundtruth))
                    fp.write("F1-score Data GroundTruth, {}\n".format(f1_score_data_groundtruth))
                    fp.write("{}".format(confmatix))
            if args.save_samples:
                dataset_utils.save_samples(save_dir + '/{}_visualization/'.format(split), recon_batches[0:1000], "_reconst", show=False)
                dataset_utils.save_samples(save_dir + '/{}_visualization/'.format(split), data[0:1000], "_original_", show=False)
            indices = []
            for i, (p_o, p_r) in enumerate(zip(pre_trained_model_data_predictions,pre_trained_model_reconst_predictions)):
                if p_o ==utils.classes['s'] and p_r ==utils.classes['t']:
                    indices.append(i)

            adv_samples = recon_batches[indices]
            adv_probs= recon_probs[indices]

            adv_data = {'samples':adv_samples, 'probs':adv_probs}

            with open(save_dir+'{}_adv_data_{}_{}.pkl'.format(split,utils.classes['s'],utils.classes['t']),"wb") as f:
                pickle.dump(adv_data,f)
            print("{}/{} {}->{} adv samples successfully generated ".format(len(indices),data.size(0),utils.classes['s'],utils.classes['t']))

    train()
    inference(split='train')
    inference(split='test')

def train_t_s():
    save_dir = utils.class_t_s_dir +'adv/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data, train_labels = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='train')
    def train():
        model = utils.ae_module.AE(dropout=args.dropout).to(utils.device)
        if os.path.exists(save_dir+'model.m'):
            model = torch.load(save_dir+'model.m')
            print("Saved model {} loaded".format(save_dir+'model.m'))
        else:
            print(" No trained model, creating one ...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=args.step_size_scheduler, gamma=args.gamma_scheduler)

        for step in range(args.steps):
            perm = torch.randperm(len(train_data))
            batch = train_data[perm[0:args.batch_size]].float().to(utils.device)
            Z,_ = pre_trained_model(batch.view(utils.get_pre_trained_model_input_shape()))
            pre_trained_model_data_probs = F.softmax(Z, dim=1)

            _,recon_batch = model(batch)

            Z,_= pre_trained_model(recon_batch.view(utils.get_pre_trained_model_input_shape()).to(utils.device))
            pre_trained_model_reconst_probs = F.softmax(Z, dim=1)
            loss = utils.loss_function_ae(recon_batch, batch, pre_trained_model_reconst_probs,
                                              pre_trained_model_data_probs, args.batch_size,
                                              utils.classes['s'], device=utils.device)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            scheduler.step()

            if step % 2000==0:
                print("Step {} Loss {}".format(step,loss.cpu().data))
        torch.save(model,save_dir+'model.m')
    def inference(split = 'test'):

        if os.path.exists(save_dir+'model.m'):
            model = torch.load(save_dir+'model.m')
            print("Saved model {} loaded".format(save_dir+'model.m'))
        else:
            print("NO trained model. Exit")
            exit(-1)
        if split =='test':
            data, labels = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='test')
        else:
            data, labels = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='train')

        with torch.no_grad():
            model.eval()
            targets = [utils.classes['t'] for _ in range(len(data))]
            ground_truth = labels.cpu().numpy()
            recon_batches, pre_trained_model_reconst_predictions,pre_trained_model_data_predictions = [],[],[]
            recon_probs = []
            for i in range(0,len(data),args.batch_size):
                start = i
                end = min(start+args.batch_size,len(data))
                recon_batch = model(data[start:end].to(utils.device))[1]

                recon_batches.append(recon_batch)
                Z,_ = pre_trained_model(data[start:end].
                                              view(utils.get_pre_trained_model_input_shape(end - start)).to(utils.device))

                pre_trained_model_data_predictions.append(np.argmax(F.softmax(Z, dim=1).cpu().numpy(),axis=1))

                Z,_ = pre_trained_model(recon_batch.
                                              view(utils.get_pre_trained_model_input_shape(end - start)).to(utils.device))

                recon_probs.append(F.softmax(Z, dim=1).cpu().numpy())
                pre_trained_model_reconst_predictions.append(np.argmax(F.softmax(Z, dim=1).cpu().numpy(),axis=1))
            recon_probs = np.concatenate(recon_probs)
            recon_batches = torch.cat(recon_batches)
            pre_trained_model_reconst_predictions = np.concatenate(pre_trained_model_reconst_predictions)
            pre_trained_model_data_predictions = np.concatenate(pre_trained_model_data_predictions)


            accuracy_score_reconst_target = accuracy_score(y_true=targets,y_pred=pre_trained_model_reconst_predictions)
            f1_score_reconst_target = f1_score(y_true=targets,y_pred=pre_trained_model_reconst_predictions,average='weighted')

            accuracy_score_data_groundtruth = accuracy_score(y_true=ground_truth,y_pred=pre_trained_model_data_predictions)
            f1_score_data_groundtruth = f1_score(y_true=ground_truth,y_pred=pre_trained_model_data_predictions,average='weighted')

            confmatix = confusion_matrix(y_true=targets,y_pred=pre_trained_model_reconst_predictions)

            with open(save_dir+'{}_adv_{}_{}.txt'.format(split,utils.classes['t'],utils.classes['s']),'w') as fp:
                    fp.write("Accurcay Reconst Target, {}\n".format(accuracy_score_reconst_target))
                    fp.write("F1-score Reconst Target, {}\n".format(f1_score_reconst_target))
                    fp.write("Accurcay Data GroundTruth, {}\n".format(accuracy_score_data_groundtruth))
                    fp.write("F1-score Data GroundTruth, {}\n".format(f1_score_data_groundtruth))
                    fp.write("{}".format(confmatix))
            if args.save_samples:
                dataset_utils.save_samples(save_dir + '/{}_visualization/'.format(split), recon_batches[0:1000], "_reconst", show=False)
                dataset_utils.save_samples(save_dir + '/{}_visualization/'.format(split), data[0:1000], "_original_", show=False)
            indices = []
            for i, (p_o, p_r) in enumerate(zip(pre_trained_model_data_predictions,pre_trained_model_reconst_predictions)):
                if p_o ==utils.classes['t'] and p_r ==utils.classes['s']:
                    indices.append(i)
            adv_samples = recon_batches[indices]
            adv_probs= recon_probs[indices]
            adv_data = {'samples':adv_samples,'probs':adv_probs}

            with open(save_dir+'{}_adv_data_{}_{}.pkl'.format(split,utils.classes['t'],utils.classes['s']),"wb") as f:
                pickle.dump(adv_data,f)
            print("{}/{} {}->{} adv samples successfully generated ".format(len(indices),data.size(0),utils.classes['t'],utils.classes['s']))
    train()
    inference(split='train')
    inference(split='test')
