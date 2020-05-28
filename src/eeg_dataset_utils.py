from itertools import zip_longest
import convert_matlab73_hdf5
from os import listdir
from os.path import join
import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np

def get_valid_ids_and_labels(trial_number, stimulus_id, stimulus_pos, trial_answer, debug=False):
    valid_trial_ids = []
    labels = []
    for experiment_number in np.unique(trial_number):
        mask = trial_number == experiment_number
        trial_pos = stimulus_pos[mask]
        trial_id = stimulus_id[mask]
        answer = trial_answer[mask][0]     
        # A valid trial has to has stimulus 1 and 3
        if 1 in trial_pos and 3 in trial_pos:            
            equal_stimulus = trial_id[trial_pos==1] == trial_id[trial_pos==3]            
            # Check if answer was correct, ommisions are considered incorrect
            label = ((answer==6) & ~equal_stimulus) | ((answer==7) & equal_stimulus)
            if debug:
                print(answer, trial_pos, trial_id, equal_stimulus, answer, label)
            valid_trial_ids.append(experiment_number)
            labels.append(label)
    return valid_trial_ids, np.array(labels)[:, 0].astype(int)


class EEG_dataset(torch.utils.data.Dataset):
    
    def __init__(self, path, transforms=None):
        # Load RMontefusco matlab files with h5py
        files = sorted(listdir(path))
        print("Loading the following files:", files)
        with h5py.File(join(path, files[0]), 'r') as f:
            matlab_dict = convert_matlab73_hdf5.recursive_dict(f)
        
        trial_info = matlab_dict['data']['trialinfo']
        session_number = trial_info[3, 0].astype(int)
        trial_number = trial_info[4, :].astype(int)
        stimulus_pos = trial_info[5, :].astype(int)
        stimulus_id = trial_info[6, :].astype(int)
        #stimulus_rot = trial_info[7, :].astype(int)
        trial_answer = trial_info[9, :].astype(int)        
        # Compute answers
        valid_trial_ids, labels = get_valid_ids_and_labels(trial_number, stimulus_id, stimulus_pos, trial_answer)
        mask_first_stimulus = (stimulus_pos == 1) & np.array([trial in set(valid_trial_ids) for trial in trial_number]) 
        mask_third_stimulus = (stimulus_pos == 3) & np.array([trial in set(valid_trial_ids) for trial in trial_number]) 
        # TODO: Make the range 200:800 and remove EOG/VOG arguments
        self.first_stimulus = matlab_dict['data']['trial'][mask_first_stimulus, 200:800, 2:]
        self.third_stimulus = matlab_dict['data']['trial'][mask_third_stimulus, 200:800, 2:]
        self.labels = labels
        # Remove EOG, VOG channels (first two)
        self.channel_names = matlab_dict['data']['cfg']['previous']['channel'][2:]
        # Times, for plotting 
        self.times = matlab_dict['data']['time'][0, 200:800]
        fs = matlab_dict["data"]['fsample']
        # Torch transforms
        self.transforms = transforms
                
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        sample = (torch.Tensor(self.first_stimulus[idx]), )
        sample += (torch.Tensor(self.third_stimulus[idx]), )
        sample += (torch.Tensor([self.labels[idx]]),)
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample

    def __len__(self):
        return len(self.labels)
            
    def plot_trial(self, idx, third_stimulus=True, figsize=(8, 8)):
        assert idx >= 0 and idx <= self.__len__(), "idx out of range"
        fig, ax = plt.subplots(6, 4, figsize=figsize, tight_layout=True)
        for i in range(6):
            for j in range(4):
                if 4*i+j < 22:
                    if third_stimulus:
                        ax[i, j].plot(self.times, self.third_stimulus[idx, :, 4*i+j])
                    else:
                        ax[i, j].plot(self.times, self.first_stimulus[idx, :, 4*i+j])
                    ax[i, j].set_title(self.channel_names[4*i+j])
        fig.show()
    
    def plot_channel(self, channel_idx, figsize=(4, 8)):
        assert channel_idx >= 0 and channel_idx < len(self.channel_names), "channel_idx out of range"
        
        fig, ax = plt.subplots(ncols=2, figsize=figsize, 
                               tight_layout=True, sharex=True, sharey=True)

        ax[0].set_ylabel(self.channel_names[channel_idx])
        color = {True: '#3182bd', False: '#de2d26'}
        for i in range(50):
            ax[0].plot(self.times, 20*i+self.first_stimulus[i, :, channel_idx], color=color[self.labels[i]])
            ax[1].plot(self.times, 20*i+self.third_stimulus[i, :, channel_idx], color=color[self.labels[i]])
        ax[0].set_xlabel('Tiempo [s]')
        ax[1].set_xlabel('Tiempo [s]')
        ax[0].set_title('Primer estímulo')
        ax[1].set_title('Tercer estímulo')

        fig.show()
        
        
        