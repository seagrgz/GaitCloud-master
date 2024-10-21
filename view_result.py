#!/usr/bin/env python
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

data_name = '[2024-10-11_07:43:50.842896]'
variance = ['00-nm', '01-nm', 'bg', 'cl', 'cr', 'ub', 'uf', 'oc', 'nt']
#variance = ['00-nm', '01-nm', '01-bg', '01-cl', '01-cr', '01-oc', '01-ub', 'uf', 'nt']
view = ['_000_', '000-far', '_045_', '_090_', '090-near', '_135_', '_180_', '180-far', '_225_', '_270_', '270-far', '_315_']
view_labels = ['000°', '045°', '090°', '135°', '180°', '225°', '270°', '315°', '*000°', '*090°', '*180°', '*270°']

mean_matrix_view = []
max_matrix_view = []

result = np.load('/home/sx-zhang/work/CNN-LSTM-master/{}/final.npy'.format(data_name), allow_pickle=True)
accuracy_val = result[0]
accuracy_train = result[1]
if len(result) > 2:
    losses = result[2]
    np.save('/home/sx-zhang/{}_loss.npy'.format(data_name), losses)
    if len(result) > 3:
        failed_record = result[3]
        failed_start = failed_record[0]

best_train = max(accuracy_train)
best_train_pos = accuracy_train.index(best_train)
print('best accuracy in train:')
print(round(best_train, 4))
keys = accuracy_val.keys()
var_keys = [name for name in keys if name in variance]
view_keys = [name for name in keys if name in view]
#print(keys)
#print(var_keys)
#print(view_keys)
best_val = {}
best_val_pos = {}
#variance
if len(var_keys) != 0:
    print('Variance evaluation results')
    for var in var_keys:
        best_val[var] = max(accuracy_val[var])
        best_val_pos[var] = accuracy_val[var].index(best_val[var])
    sum_val = [sum(x) for x in zip(*(accuracy_val[name] for name in var_keys))]
    best_mean_pos = sum_val.index(max(sum_val))
    print('best accuracy in test:')
    print(best_val)
    for var in var_keys:
        print('accuracy at {} best:'.format(var))
        print('Epoch:', best_val_pos[var], 'train:', round(accuracy_train[best_val_pos[var]], 4), end = ' ')
        for attr in var_keys:
            print(attr, ':', round(accuracy_val[attr][best_val_pos[var]], 4), end = ' ')
        print()
    print('best mean accuracy in test')
    print('Epoch:', best_mean_pos, 'train:', round(accuracy_train[best_mean_pos], 4), end = ' ')
    for attr in var_keys:
        print(attr, ':', round(accuracy_val[attr][best_mean_pos], 4), end = ' ')
    print('\n ')
    #if len(result) > 3:
    #    best_var_failed = failed_record[best_mean_pos-failed_start][0]
#view
if len(view_keys) != 0:
    #print('Cross view evaluation results')
    sum_val = {}
    if type(accuracy_val[view_keys[0]]) is dict:
        for gallery in view_keys:
            best_max = []
            for vp in view_keys:
                best_max.append(max(accuracy_val[gallery][vp]))
            sum_val[gallery] = [sum(x) for x in zip(*(accuracy_val[gallery][name] for name in view_keys))]
            max_matrix_view.append(best_max)

        sum_total_val = [sum(x) for x in zip(*(sum_val[name] for name in view_keys))]
        best_mean_pos = sum_total_val.index(max(sum_total_val))
        if 'overall' in keys:
            if len(accuracy_val['overall']) == 2:
                print('Epoch: {}/{}, overall: {}'.format(best_mean_pos, accuracy_val['overall'][1], round(accuracy_val['overall'][0], 4)*100))
            elif len(accuracy_val['overall']) > 2:
                print('Epoch: {}, overall: {}'.format(best_mean_pos, round(accuracy_val['overall'][best_mean_pos], 4)))
        mean_matrix_view = []
        for gallery in view_keys:
            best_mean = []
            for vp in view_keys:
                best_mean.append(accuracy_val[gallery][vp][best_mean_pos])
            mean_matrix_view.append(best_mean)
        #if len(result) > 3:
        #    best_view_failed = failed_record[best_mean_pos-failed_start][1]
            #print(best_view_failed['270-far']['090-near']) #[gallery][probe]
    else:
        for var in view_keys:
            best_val[var] = max(accuracy_val[var])
            best_val_pos[var] = accuracy_val[var].index(best_val[var])
        sum_val = [sum(x) for x in zip(*(accuracy_val[name] for name in view_keys))]
        best_mean_pos = sum_val.index(max(sum_val))

    #create confusion matrix
    #baseline = np.asarray([
    #    [83,76,67,50,45,69,77,70,65,50,43,70],
    #    [68,85,70,78,65,83,73,79,65,75,69,66],
    #    [46,72,87,71,46,70,75,69,49,79,47,70],
    #    [66,78,70,87,68,78,74,83,68,75,64,68],
    #    [79,68,51,73,83,68,51,69,75,55,73,47],
    #    [63,81,69,76,64,86,76,77,62,73,68,67],
    #    [45,70,75,71,47,74,84,72,47,77,47,76],
    #    [66,76,66,80,62,78,75,85,72,71,60,66],
    #    [76,64,47,69,69,64,50,73,81,49,57,46],
    #    [47,73,77,72,47,72,80,70,48,83,49,74],
    #    [68,68,49,65,72,71,50,63,58,50,82,47],
    #    [36,59,63,56,37,61,74,58,37,69,37,80]])
    baseline = np.asarray([
        [92,89,84,87,87,87,86,90,92,86,87,84],
        [91,93,88,90,87,91,90,92,89,92,89,88],
        [83,87,94,89,84,86,87,87,83,90,84,85],
        [89,92,89,94,92,91,90,92,89,92,91,88],
        [89,88,84,90,93,90,86,88,88,87,91,82],
        [86,91,85,91,90,92,90,90,86,90,91,87],
        [83,87,87,87,83,87,92,90,85,88,83,89],
        [89,89,85,91,86,89,90,92,89,89,85,87],
        [90,87,82,85,85,84,85,89,92,86,83,82],
        [86,90,90,90,85,88,91,89,86,93,85,87],
        [86,86,82,89,89,89,85,86,83,85,93,83],
        [76,80,78,78,75,81,82,80,74,83,74,89]])
    base_mean = np.mean(baseline)
    lose_indices = np.nonzero(baseline > np.asarray(mean_matrix_view)*100)
    array_labels = np.array(view_labels)
    lose_pairs = [list(pair) for pair in np.column_stack((array_labels[lose_indices[0]], array_labels[lose_indices[1]])) if pair[0]!=pair[1]]
    print('lose pairs: ', lose_pairs)
    #print(view_keys)
    #print(view_labels)

    if type(accuracy_val[view_keys[0]]) is dict:
        mean_matrix_view = (np.asarray(mean_matrix_view)*100).astype(int)
        max_matrix_view = (np.asarray(max_matrix_view)*100).astype(int)
        #calculate overall accuracy
        masked = mean_matrix_view.copy()
        np.fill_diagonal(masked, 0)
        overall = masked.sum()/(masked.shape[0]*(masked.shape[1]-1))
        vmin = 50 #min(mean_matrix_view.min(), max_matrix_view.min())
        vmax = 100 #max(mean_matrix_view.max(), max_matrix_view.max())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        sn.heatmap(mean_matrix_view, annot=True, fmt='d', cmap='viridis', ax=axes[0], vmin=vmin, vmax=vmax, xticklabels=view_labels, yticklabels=view_labels)
        axes[0].set_title('Uniformed best accuracy ({})'.format(round(overall, 2)))
        axes[0].set_xlabel('probe')
        axes[0].set_ylabel('gallery')
        axes[0].tick_params(axis='x', rotation=45)
        sn.heatmap(max_matrix_view, annot=True, fmt='d', cmap='viridis', ax=axes[1], vmin=vmin, vmax=vmax, xticklabels=view_labels, yticklabels=view_labels)
        axes[1].set_title('Independent best accuracy')
        axes[1].set_xlabel('probe')
        axes[1].set_ylabel('gallery')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig('{}view matices.png'.format(data_name))
        plt.close()

        fig, ax = plt.subplots(figsize=(6,5), dpi=300)
        sn.heatmap(baseline, annot=True, fmt='d', cmap='viridis', vmin=vmin, vmax=vmax, xticklabels=view_labels, yticklabels=view_labels)
        ax.set_xlabel('probe')
        ax.set_ylabel('gallery')
        ax.set_title('Uniformed best accuracy ({})'.format(base_mean))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('baseline.png')
        plt.close()
    else:
        pass
