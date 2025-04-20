import argparse
import math
import pickle
from tqdm import tqdm
import sys
import json
sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization2d


# joints distrubution
joints = ['head', 'nose' ,'Neck' ,'Chest' ,'Mhip' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet = {'nose': 12, 'head':12, 'Lsho':11, 'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}



max_body_true = 2
max_body = 2
num_joint = 17
max_frame = 51000
num_classes = 51

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as json_data:
        skeleton_sequence = json.load(json_data)

    return skeleton_sequence


def get_nonzero_std(s): 
    index = s.sum(-1).sum(-1) != 0  
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std()
    else:
        s = 0
    return s
    
def normalize_screen_coordinates( X, w, h):
    assert X.shape[-1] == 2
    zeros=np.where(X==0)
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    center= X/w*2 - [1, h/w]
    center[zeros]=0
    return center


def read_xyz(file, max_body=2, num_joint=17):  
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, len(seq_info['frames']), num_joint, 2))
    nb_frames = len(seq_info['frames'])
    for n, f in enumerate(seq_info['frames']):
        if len(f) != 0:
            for m, b in enumerate(f):
                if m < max_body:
                    for j,k in enumerate(joints):
                        if k == 'Mhip':
                            data[m, n, j, :] = [ (b['pose2d'][4] + b['pose2d'][5])/2, (b['pose2d'][17] + b['pose2d'][18])/2 ]
                        elif k == 'Neck':
                            data[m, n, j, :] = [ (b['pose2d'][10] + b['pose2d'][11])/2, (b['pose2d'][23] + b['pose2d'][24])/2 ]  
                        elif k == 'Chest':
                            data[m, n, j, :] = [ (b['pose2d'][4] + b['pose2d'][5] + b['pose2d'][10] + b['pose2d'][11])/4, (b['pose2d'][17] + b['pose2d'][18] + b['pose2d'][23] + b['pose2d'][24])/4 ]  
                        else:
                            data[m, n, j, :] = [ b['pose2d'][lcrnet[k]], b['pose2d'][lcrnet[k] + 13] ]
                    data[m, n, 1, :]=(data[m, n, 0, :]+data[m, n, 2, :])/2
                else:
                    pass

    # select the max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    
    
    # centralization
    for i in range(data.shape[0]):
        keypoints=data[i,:,:,:2]
        keypoints = normalize_screen_coordinates(keypoints[..., :2], w=640, h=480)
        data[i,:,:,:2]=keypoints


    data = data.transpose(3, 1, 2, 0)
    return data, nb_frames


def gendata(data_path, out_path, split_path, benchmark='xview', part='eval'):
    with open(split_path, 'r') as f:
        data_split = json.load(f)
    ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        vid = filename[:-5]
        istraining = data_split[vid]['subset'] == 'training'
        istesting = data_split[vid]['subset'] == 'testing'
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = istesting
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename) 
    
    fp = np.zeros((len(sample_name), 2, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name[:])):
        vid = s[:-5]
        data, nb_frame = read_xyz(os.path.join(data_path, s), max_body = max_body, num_joint = num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
        fp[i, -1, -1, -1, -1] = nb_frame

        label = np.zeros((max_frame, num_classes), np.float32)
        fps = float(nb_frame/float(data_split[vid]['duration']))
        for ann in data_split[vid]['actions']:
            for fr in range(0, nb_frame, 1):
                # print (fr,num_feat,fps)
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[fr, ann[0]] = 1 # bi
        sample_label.append((label, data_split[vid]['duration'])) 

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    #fp = pre_normalization2d(fp)
  
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
  
    print(len(sample_label), out.shape[0])


        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smarthome Data Converter.')
    parser.add_argument('--data_path', default='./data/tsu_raw/tsu_skeletons/')
    parser.add_argument('--split_path',
                        default='./data/tsu_raw/smarthome_CS_51.json')

    parser.add_argument('--out_folder', default='../data/tsu/')

    benchmark = ['xsub']#, 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()
    print('skeleton path: ', arg.data_path)
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.split_path,
                benchmark=b,
                part=p)
