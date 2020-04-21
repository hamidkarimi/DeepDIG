import os
import numpy as np
import torch
import random
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report
import torch.nn.functional as F
import sys
import json
from tqdm import tqdm
import importlib
import pickle
from sklearn import svm

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

def get_pre_trained_model_features(data):
    features = []
    with torch.no_grad():
        for i in range(0,len(data),args.batch_size):
            start = i
            end = min(start+args.batch_size,len(data))
            _,feats = pre_trained_model(data[start:end].to(utils.device))
            feats = feats.cpu().numpy()
            features.append(feats)
        features = np.concatenate(features)
    return features

def linearity_metrics():
    train_data_s, train_labels_s = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='train')
    Y_train_s = train_labels_s.cpu().numpy()
    X_train_s,_ = utils.get_pre_trained_model_features_predictions(train_data_s,pre_trained_model)

    test_data_s, test_labels_s = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='test')
    Y_test_s = test_labels_s.cpu().numpy()
    X_test_s,_ = utils.get_pre_trained_model_features_predictions(test_data_s,pre_trained_model)
    ##############################
    train_data_t, train_labels_t = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='train')
    Y_train_t = train_labels_t.cpu().numpy()
    X_train_t,_ = utils.get_pre_trained_model_features_predictions(train_data_t,pre_trained_model)

    test_data_t, test_labels_t = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='test')
    Y_test_t = test_labels_t.cpu().numpy()
    X_test_t,_ = utils.get_pre_trained_model_features_predictions(test_data_t,pre_trained_model)


    with open(utils.class_s_t_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'rb') as f:
        borderline_data_s= pickle.load(f)
    X_borderline_s = np.squeeze(borderline_data_s['success_samples']['feats_borderline'])
    Y_borderline_s = np.array([utils.classes['s'] for _ in range(len(X_borderline_s))])

    with open(utils.class_t_s_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'rb') as f:
        borderline_data_t= pickle.load(f)
    X_borderline_t = np.squeeze(borderline_data_t['success_samples']['feats_borderline'])
    Y_borderline_t = np.array([utils.classes['t'] for _ in range(len(X_borderline_t))])

    svm_model = svm.LinearSVC(C=1.0, max_iter=10000)

    X_train = np.concatenate((X_train_s,X_train_t))
    Y_train = np.concatenate((Y_train_s,Y_train_t))

    X_test = np.concatenate((X_test_s,X_test_t))
    Y_test = np.concatenate((Y_test_s,Y_test_t))

    X_borderline = np.concatenate((X_borderline_s,X_borderline_t))
    Y_borderline = np.concatenate((Y_borderline_s,Y_borderline_t))

    svm_model.fit(X_train, Y_train)
    w_norm = np.linalg.norm(svm_model.coef_)
    results = {}

    pred = svm_model.predict(X_train)
    all_distances = svm_model.decision_function(X_train)/w_norm
    max, min = np.max(all_distances), np.mean(all_distances)
    m1 = np.mean(np.abs(all_distances))/(1+np.mean(np.abs(all_distances)))
    results['train_abs_mean_all_distances'] = m1
    results['train_max_all_distance'] = max
    results['train_min_all_distance'] = min
    erroneous =X_train[[i for i, (a,b) in enumerate(zip(pred, Y_train)) if a!=b]]
    if len(erroneous) >0:
        erroneous_distances = svm_model.decision_function(erroneous)/w_norm
        max, min = np.max(erroneous_distances), np.mean(erroneous_distances)
        m2 = np.mean(np.abs(erroneous_distances))/(1+np.mean(np.abs(erroneous_distances)))
        results['train_abs_erroneous_distances'] = m2
        results['train_max_erroneous_distances_distance'] = max
        results['train_min_erroneous_distances_distance'] = min

    results['train_classification_report']= classification_report(y_true=Y_train, y_pred = pred)
    results['train_acc'] = accuracy_score(y_true=Y_train, y_pred = pred)

    pred = svm_model.predict(X_test)
    all_distances = svm_model.decision_function(X_test)/w_norm
    max, min = np.max(all_distances), np.mean(all_distances)
    m1 = np.mean(np.abs(all_distances))/(1+np.mean(np.abs(all_distances)))
    results['test_abs_mean_all_distances'] = m1
    results['test_max_all_distance'] = max
    results['test_min_all_distance'] = min
    erroneous =X_test[[i for i, (a,b) in enumerate(zip(pred, Y_test)) if a!=b]]
    if len(erroneous) >0:
        erroneous_distances = svm_model.decision_function(erroneous)/w_norm
        max, min = np.max(erroneous_distances), np.mean(erroneous_distances)
        m2 = np.mean(np.abs(erroneous_distances))/(1+np.mean(np.abs(erroneous_distances)))
        results['test_abs_erroneous_distances'] = m2
        results['test_max_erroneous_distances_distance'] = max
        results['test_min_erroneous_distances_distance'] = min

    results['test_classification_report']= classification_report(y_true=Y_test, y_pred = pred)
    results['test_acc'] = accuracy_score(y_true=Y_test, y_pred = pred)
    pred = svm_model.predict(X_borderline)

    all_distances = svm_model.decision_function(X_borderline)/w_norm
    max, min = np.max(all_distances), np.mean(all_distances)
    m1 = np.mean(np.abs(all_distances))/(1+np.mean(np.abs(all_distances)))
    results['borderline_abs_mean_all_distances'] = m1
    results['borderline_max_all_distance'] = max
    results['borderline_min_all_distance'] = min
    erroneous =X_borderline[[i for i, (a,b) in enumerate(zip(pred, Y_borderline)) if a!=b]]
    if len(erroneous) >0:
        erroneous_distances = svm_model.decision_function(erroneous)/w_norm
        max, min = np.max(erroneous_distances), np.mean(erroneous_distances)
        m2 = np.mean(np.abs(erroneous_distances))/(1+np.mean(np.abs(erroneous_distances)))
        results['borderline_abs_erroneous_distances'] = m2
        results['borderline_max_erroneous_distances_distance'] = max
        results['borderline_min_erroneous_distances_distance'] = min

    results['borderline_classification_reposrt']= classification_report(y_true=Y_borderline, y_pred = pred)
    results['borderline_acc'] = accuracy_score(y_true=Y_borderline, y_pred = pred)
    r = json.dumps(results)
    f = open(utils.working_dir+'linearity_metrics.json',"w")
    f.write(r)
    f.close()
    #print("Linearity metrics extrcated for {}".format(utils.classes))

def sample_trajectory_points(x0,x1):
    Xts=[]

    for i in  range(args.num_samples_trajectory):
        t= i/args.num_samples_trajectory
        Xts.append(t*x0+(1-t)*x1)
    Xts=torch.stack(Xts)
    return Xts.to(utils.device)

def trajectory_smoothness(probs):
    X =0
    for i in range(len(probs)-1):
        if probs[i] !=probs[i+1]:
            X+=1
    return X/len(probs)

def get_trajectory_smoothness(samples):
    S = []
    l= [k for k in range(len(samples))]
    for i in tqdm(range(len(samples))):
        l.remove(i)
        indices= random.sample(l,args.num_samples_trajectory)
        l.append(i)
        for j in indices:
            Xts = sample_trajectory_points(samples[i],samples[j])
            Z,_ = pre_trained_model(Xts.view(utils.get_pre_trained_model_input_shape(Xts.size(0))))
            P = np.argmax(F.softmax(Z, dim=1).detach().cpu().numpy(),axis=1)
            smoothness = trajectory_smoothness(P)
            S.append(smoothness)
    S = np.array(S)
    return np.mean(S)

def trajectory_metrics():
    with open(utils.class_s_t_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'rb') as f:
        borderline_data_s= pickle.load(f)
    samples = borderline_data_s['success_samples']['samples']
    smoothness_s_t=get_trajectory_smoothness(samples)


    with open(utils.class_t_s_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'rb') as f:
        borderline_data_t= pickle.load(f)
    samples = borderline_data_t['success_samples']['samples']
    smoothness_t_s=get_trajectory_smoothness(samples)

    results= {"smoothness_s_t":smoothness_s_t,"smoothness_t_s":smoothness_t_s}
    r = json.dumps(results)
    f = open(utils.working_dir+'trajectory_metrics.json',"w")
    f.write(r)
    f.close()




