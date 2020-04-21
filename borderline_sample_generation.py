import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import sys
import importlib
import pickle
from tqdm import tqdm
from DeepDIGCode import config
from DeepDIGCode import utils
distance_function= np.linalg.norm
import random
args = config.args

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model)
sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset)

dataset_utils = importlib.import_module("DeepDIGCode.PreTrainedModels." + args.dataset + ".utils")
from model import *
pre_trained_model =torch.load(args.project_dir+'PreTrainedModels/'+
                              args.dataset+'/'+ args.pre_trained_model+'/pre_trained_model.m').to(utils.device)

pre_trained_model.eval()


def deepdig_borderline_binary_search (pairs,class1,class2):
    failed_samples = {"samples":[],"probs_x1":[],"probs_x2":[],"probs_borderline":[],"attempts":[],"feats_borderline":[]}
    success_samples = {"samples":[],"probs_x1":[],"probs_x2":[],"probs_borderline":[],"attempts":[],"feats_borderline":[]}
    with torch.no_grad():
        for  sample1, sample2 in tqdm(pairs):

            x1=sample1
            x2=sample2
            attempts = 0
            while (True):
                middle_sample = (x1+x2)/2
                attempts += 1
                Z,_ = pre_trained_model(x1.unsqueeze(0).to(utils.device))
                probs_x1 = F.softmax(Z, dim=1).cpu().numpy().flatten()
                pred_x1 =np.argmax(probs_x1)

                Z,_ = pre_trained_model(x2.unsqueeze(0).to(utils.device))
                probs_x2 = F.softmax(Z, dim=1).cpu().numpy().flatten()
                pred_x2 = np.argmax(probs_x2)

                Z,feats_middle = pre_trained_model(middle_sample.unsqueeze(0).to(utils.device))
                probs_middle = F.softmax(Z, dim=1).cpu().numpy().flatten()
                pred_middle = np.argmax(probs_middle)
                feats_middle = feats_middle.cpu().detach().numpy()
                if pred_middle != class1 and pred_middle != class2:

                    failed_samples['samples'].append(middle_sample)
                    failed_samples['probs_x1'].append(probs_x1)
                    failed_samples['probs_x2'].append(probs_x2)
                    failed_samples['probs_borderline'].append(probs_middle)
                    failed_samples['attempts'].append(attempts)
                    failed_samples['feats_borderline'].append(feats_middle)
                    break

                if np.abs(probs_middle[class1]-probs_middle[class2]) < args.middle_point_threshold:

                    success_samples['samples'].append(middle_sample)
                    success_samples['probs_x1'].append(probs_x1)
                    success_samples['probs_x2'].append(probs_x2)
                    success_samples['probs_borderline'].append(probs_middle)
                    success_samples['attempts'].append(attempts)
                    success_samples['feats_borderline'].append(feats_middle)
                    break

                if pred_middle == pred_x1:
                    x1 = middle_sample
                if pred_middle == pred_x2:
                    x2 = middle_sample


        if len(success_samples['samples'])>0:
            success_samples['samples'] = torch.stack(success_samples['samples'])
            success_samples['probs_x1'] = np.array(success_samples['probs_x1'])
            success_samples['probs_x2']= np.array(success_samples['probs_x2'])
            success_samples['probs_borderline']= np.array(success_samples['probs_borderline'])
            success_samples['attempts']= np.array(success_samples['attempts'])
            success_samples['feats_borderline'] =np.array(success_samples['feats_borderline'])
        if len(failed_samples['samples'])>0:
            failed_samples['samples'] = torch.stack(failed_samples['samples'])
            failed_samples['probs_x1'] = np.array(failed_samples['probs_x1'])
            failed_samples['probs_x2']= np.array(failed_samples['probs_x2'])
            failed_samples['probs_borderline']= np.array(failed_samples['probs_borderline'])
            failed_samples['attempts']= np.array(failed_samples['attempts'])
            failed_samples['feats_borderline'] =np.array(failed_samples['feats_borderline'])
        success_rate = -1
        average_attempts_success = -1
        average_attempts_fail = -1
        average_attempts_total = -1

        if len(success_samples['samples'])==0 and len(failed_samples['samples'])==0:
            pass
        elif len(success_samples['samples'])==0 and len(failed_samples['samples'])>0:
            average_attempts_fail = np.mean(failed_samples['attempts'])
            average_attempts_total = average_attempts_fail
        elif len(success_samples['samples'])>0 and len(failed_samples['samples']) == 0:
            success_rate = 1.0
            average_attempts_success = np.mean(success_samples['attempts'])
            average_attempts_total = average_attempts_success
        elif len(success_samples['samples'])>0 and len(failed_samples['samples'])>0:
            success_rate = len(success_samples['samples'])/(len(success_samples['samples'])+len(failed_samples['samples']))
            average_attempts_success = np.mean(success_samples['attempts'])
            average_attempts_fail = np.mean(failed_samples['attempts'])
            average_attempts_total = np.mean(np.concatenate((success_samples['attempts'],failed_samples['attempts'])))

        return {"success_samples": success_samples,
                "failed_samples":failed_samples,
                "success_rate":success_rate,
                "avg_attempt_success":average_attempts_success,
                "avg_attempt_fail":average_attempts_fail,
                "avg_attempt_total":average_attempts_total}


def deepdig_borderline_samples_s_t():
    data_dir  = utils.class_s_t_dir +'adv_of_adv/'
    with open(data_dir+'adv_of_adv_data_{}_{}_{}.pkl'.format(utils.classes['s'],utils.classes['t'],utils.classes['s']),"rb") as f:
        data = pickle.load(f)

    borderline_samples = deepdig_borderline_binary_search(zip(data['adv_samples'],data['adv_of_adv_samples']),
                                                           utils.classes['s'],
                                                           utils.classes['t'])

    with open(utils.class_s_t_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'wb') as f:
        pickle.dump(borderline_samples, f)
        print("{} saved".format(utils.class_s_t_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold)))
    if args.save_samples and len(borderline_samples['success_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.class_s_t_dir +'/borderline_samples_deepdig_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['success_samples']['samples'],
                                   "success_borderline", show=False)
    if args.save_samples and len(borderline_samples['failed_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.class_s_t_dir +'/borderline_samples_deepdig_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['failed_samples']['samples'],
                                   "failed_borderline", show=False)

    print("Class {}->{} \n\tsuccess rate {}\n\tavg attempts success {}\n\tavg attempts fail {}\n\tavg attempts total {}"
          .format(utils.classes['s'], utils.classes['t'],
                  borderline_samples['success_rate'],
                  borderline_samples['avg_attempt_success'],
                  borderline_samples['avg_attempt_fail'],
                  borderline_samples['avg_attempt_total']))
    print('*' * 100)




def deepdig_borderline_samples_t_s():
    data_dir  = utils.class_t_s_dir +'adv_of_adv/'
    with open(data_dir+'adv_of_adv_data_{}_{}_{}.pkl'.format(utils.classes['t'],utils.classes['s'],utils.classes['t']),"rb") as f:
        data = pickle.load(f)

    borderline_samples = deepdig_borderline_binary_search(zip(data['adv_samples'],data['adv_of_adv_samples']),
                                                           utils.classes['t'],
                                                           utils.classes['s'])

    with open(utils.class_t_s_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold), 'wb') as f:
        pickle.dump(borderline_samples, f)
        print("{} saved".format(utils.class_t_s_dir + 'borderline_deepdig_{}.pkl'.format(args.middle_point_threshold)))
    if args.save_samples and len(borderline_samples['success_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.class_t_s_dir +'/borderline_samples_deepdig_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['success_samples']['samples'],
                                   "success_borderline", show=False)
    if args.save_samples and len(borderline_samples['failed_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.class_t_s_dir +'/borderline_samples_deepdig_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['failed_samples']['samples'],
                                   "failed_borderline", show=False)

    print("Class {}->{} \n\tsuccess rate {}\n\tavg attempts success {}\n\tavg attempts fail {}\n\tavg attempts total {}"
          .format(utils.classes['t'], utils.classes['s'],
                  borderline_samples['success_rate'],
                  borderline_samples['avg_attempt_success'],
                  borderline_samples['avg_attempt_fail'],
                  borderline_samples['avg_attempt_total']))
    print('*' * 100)

def random_pair_borderline_search():
    train_data_s, _ = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='train')
    _,prediction_s = utils.get_pre_trained_model_features_predictions(train_data_s,pre_trained_model)
    indices_s = np.where(prediction_s==utils.classes['s'])[0]
    random.shuffle(indices_s)

    train_data_t, _ = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='train')
    _,prediction_t = utils.get_pre_trained_model_features_predictions(train_data_t,pre_trained_model)
    indices_t = np.where(prediction_t==utils.classes['t'])[0]
    random.shuffle(indices_t)
    a=min(len(indices_s),len(indices_t))

    indices_s =indices_s[0:a]
    indices_t = indices_t[0:a]

    borderline_samples = deepdig_borderline_binary_search(zip(train_data_s[indices_s],train_data_t[indices_t]),
                                                           utils.classes['s'],
                                                           utils.classes['t'])
    with open(utils.working_dir + 'random_borderline_{}.pkl'.format(args.middle_point_threshold), 'wb') as f:
        pickle.dump(borderline_samples, f)
        print("{} saved".format(utils.working_dir + 'random_borderline_{}.pkl'.format(args.middle_point_threshold)))
    if args.save_samples and len(borderline_samples['success_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.working_dir +'/random_borderline_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['success_samples']['samples'],
                                   "success_borderline", show=False)
    if args.save_samples and len(borderline_samples['failed_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.working_dir +'/random_borderline_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['failed_samples']['samples'],
                                   "failed_borderline", show=False)

    print("\n\tsuccess rate {}\n\tavg attempts success {}\n\tavg attempts fail {}\n\tavg attempts total {}"
          .format(borderline_samples['success_rate'],
                  borderline_samples['avg_attempt_success'],
                  borderline_samples['avg_attempt_fail'],
                  borderline_samples['avg_attempt_total']))
    print('*' * 100)



def embedding_nearest_pair_borderline_search():
    train_data_s, _ = dataset_utils.get_class_specific_data(Class=utils.classes['s'], split='train')
    feats_s,prediction_s = utils.get_pre_trained_model_features_predictions(train_data_s,pre_trained_model)
    indices_s = np.where(prediction_s==utils.classes['s'])[0]
    train_data_s = train_data_s[indices_s]
    feats_s = feats_s[indices_s]

    train_data_t, _ = dataset_utils.get_class_specific_data(Class=utils.classes['t'], split='train')
    feats_t,prediction_t = utils.get_pre_trained_model_features_predictions(train_data_t,pre_trained_model)
    indices_t = np.where(prediction_t==utils.classes['t'])[0]
    train_data_t = train_data_t[indices_t]
    feats_t = feats_t[indices_t]

    print(feats_t.shape, feats_s.shape)

    data= feats_t

    indices =[]
    for f in tqdm(feats_s):
        F = f
        A = np.transpose(np.repeat(F.reshape(len(F),1),len(data),axis=1))
        d= distance_function(A-data)
        j=np.argmin(d)
        indices.append(j)

    pairs = zip(train_data_s, train_data_t[indices])
    borderline_samples = deepdig_borderline_binary_search(pairs,utils.classes['s'], utils.classes['t'])
    with open(utils.working_dir + 'embedding_nearest_borderline_{}.pkl'.format(args.middle_point_threshold), 'wb') as f:
        pickle.dump(borderline_samples, f)
        print("{} saved".format(utils.working_dir + 'embedding_nearest_borderline_{}.pkl'.format(args.middle_point_threshold)))
    if args.save_samples and len(borderline_samples['success_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.working_dir +'/embedding_nearest_borderline_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['success_samples']['samples'],
                                   "success_borderline", show=False)
    if args.save_samples and len(borderline_samples['failed_samples']['samples']) > 0:
        dataset_utils.save_samples(utils.working_dir +'/embedding_nearest_borderline_{}/'.
                                   format(args.middle_point_threshold),
                                   borderline_samples['failed_samples']['samples'],
                                   "failed_borderline", show=False)

    print("\n\tsuccess rate {}\n\tavg attempts success {}\n\tavg attempts fail {}\n\tavg attempts total {}"
          .format(borderline_samples['success_rate'],
                  borderline_samples['avg_attempt_success'],
                  borderline_samples['avg_attempt_fail'],
                  borderline_samples['avg_attempt_total']))
    print('*' * 100)
