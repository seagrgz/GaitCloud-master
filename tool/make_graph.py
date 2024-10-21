#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

#evaluation setup
data_root = '/home/sx-zhang/work/CNN-LSTM-master/paperdata/'
eval_range = 20

feature = ['Intensity', 'XYZ', 'Depth', 'Fix', 'Random']
repro = ['dir1', 'dir2', 'dir3', 'dir4', 'combined']
label = ['Set90', 'Set0', 'Set135', 'Set45', 'combined']
label_con = ['ins', 'xyz', 'depth', 'fix', 'random']
method_list = ['cnn+lstm4', 'cnn+lstm1', 'cnn+diff+lstm1', 'cnn']
method_name = ['LSTM4', 'LSTM1', 'Differ-LSTM', 'Mean']
part_line = ['[3, 4, 5]','[4, 6, 10]','[5, 9, 10]','[6, 7, 10]','[3, 5, 10]','[8, 9, 10]']
volume_list = ['double', 'float32', 'float16']

result = {'dir1':[[],[],[]], 'dir2':[[],[],[]], 'dir3':[[],[],[]], 'dir4':[[],[],[]], 'combined':[[],[],[]]}

#result format: accuracy, accu_mean, accu_error, conver_e
def eval_record(name, subset, split, epoch):
    global data_root
    data_path = os.path.join(data_root, '{}/{}-{}.npy'.format(split, name, subset))
    #print(data_path)
    result = np.load(data_path, allow_pickle=True)
    accu = result[0]
    #best_accu = sum(accu[epoch-mean_range:epoch])/mean_range
    #best_var = np.var(accu[epoch-mean_range:epoch])
    step_accu = []
    for step in range(eval_range,epoch):
        step_accu.append(np.mean(accu[step-eval_range:step]))
    best_accu = max(step_accu)
    best_pos = step_accu.index(best_accu)
    best_var = np.std(accu[best_pos:best_pos+eval_range])

    for pos, step_accuracy in enumerate(step_accu):
        step_var = np.std(accu[pos:pos+eval_range])
        if step_accuracy > best_accu-0.02 and step_var < best_var+0.01:
            conver_e = pos+eval_range
            break
    return best_accu, best_var, conver_e, accu

def eval_con_record(feature, method, subset, split, epoch):
    global data_root
    data_path = os.path.join(data_root, '{}/{}-{}-{}.npy'.format(split, feature, method, subset))
    #print(data_path)
    result = np.load(data_path, allow_pickle=True)
    accu = result[0]
    #best_accu = sum(accu[epoch-mean_range:epoch])/mean_range
    #best_var = np.var(accu[epoch-mean_range:epoch])
    step_accu = []
    for step in range(eval_range,epoch):
        step_accu.append(np.mean(accu[step-eval_range:step]))
    best_accu = max(step_accu)
    best_pos = step_accu.index(best_accu)
    best_var = np.std(accu[best_pos:best_pos+eval_range])

    for pos, step_accuracy in enumerate(step_accu):
        step_var = np.std(accu[pos:pos+eval_range])
        if step_accuracy > best_accu-0.02 and step_var < best_var+0.01:
            conver_e = pos+eval_range
            break
    return best_accu, best_var, conver_e, accu

def plot_feature(result):
    #linestyle = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
    plt.figure(figsize=(6,4))
    bar_c = np.array([1, 3, 5, 7, 9])
    offset = 0
    for i, subset in enumerate(repro):
        plt.bar(bar_c-0.4+offset, result[subset][0], 0.2, yerr=[err for err in result[subset][1]], capsize=3, label=label[i])
        offset += 0.2

    plt.ylim([0.6,1.1])
    plt.legend(loc='lower right')
    plt.xticks(bar_c, feature)
    plt.xlabel('input value')
    plt.ylabel('best accuracy')
    plt.savefig('f_select_best_accuracy.png')

    plt.figure(figsize=(6,4))
    bar_c = np.array([1, 3, 5, 7, 9])
    offset = 0
    for i, subset in enumerate(repro):
        plt.bar(bar_c-0.4+offset, result[subset][2], 0.2, label=label[i])
        offset += 0.2

    plt.ylim([0,1000])
    plt.legend(loc='upper right')
    plt.xticks(bar_c, feature)
    plt.xlabel('input value')
    plt.ylabel('convergence time')
    plt.savefig('f_select_convergence time.png')

def plot_line(result):
    bar_c = np.arange(0.5,10.5,1)

    #dir1
    fig_dir1, ax1 = plt.subplots(figsize=(5,2.5))
    ax1.bar(bar_c, result['dir1'][0], 0.6, yerr=[err for err in result['dir1'][1]], capsize=4, color='forestgreen')
    ax1.set_xlabel('number of lines')
    ax1.set_ylabel('best_accuracy')
    ax1.set_ylim([0.6,1.1])
    plt.xticks(bar_c, range(16,6,-1))

    ax2 = ax1.twinx()
    ax2.plot(bar_c, result['dir1'][2], linewidth=4, color='darkorange')
    ax2.set_ylim([1,1000])
    ax2.set_ylabel('convergence time')

    fig_dir1.tight_layout()
    plt.savefig('lineselect-dir1.png')

    #dir2
    fig_dir2, ax1 = plt.subplots(figsize=(5,2.5))
    ax1.bar(bar_c, result['dir2'][0], 0.6, yerr=[err for err in result['dir2'][1]], capsize=4, color='forestgreen')
    ax1.set_xlabel('number of lines')
    ax1.set_ylabel('best_accuracy')
    ax1.set_ylim([0.6,1.1])
    plt.xticks(bar_c, range(16,6,-1))

    ax2 = ax1.twinx()
    ax2.plot(bar_c, result['dir2'][2], linewidth=4, color='darkorange')
    ax2.set_ylim([1,1000])
    ax2.set_ylabel('convergence time')

    fig_dir2.tight_layout()
    plt.savefig('lineselect-dir2.png')

    #dir3
    fig_dir3, ax1 = plt.subplots(figsize=(5,2.5))
    ax1.bar(bar_c, result['dir3'][0], 0.6, yerr=[err for err in result['dir3'][1]], capsize=4, color='forestgreen')
    ax1.set_xlabel('number of lines')
    ax1.set_ylabel('best_accuracy')
    ax1.set_ylim([0.6,1.1])
    plt.xticks(bar_c, range(16,6,-1))

    ax2 = ax1.twinx()
    ax2.plot(bar_c, result['dir3'][2], linewidth=4, color='darkorange')
    ax2.set_ylim([1,1000])
    ax2.set_ylabel('convergence time')

    fig_dir3.tight_layout()
    plt.savefig('lineselect-dir3.png')

    #dir4
    fig_dir4, ax1 = plt.subplots(figsize=(5,2.5))
    ax1.bar(bar_c, result['dir4'][0], 0.6, yerr=[err for err in result['dir4'][1]], capsize=4, color='forestgreen')
    ax1.set_xlabel('number of lines')
    ax1.set_ylabel('best_accuracy')
    ax1.set_ylim([0.6,1.1])
    plt.xticks(bar_c, range(16,6,-1))

    ax2 = ax1.twinx()
    ax2.plot(bar_c, result['dir4'][2], linewidth=4, color='darkorange')
    ax2.set_ylim([1,1000])
    ax2.set_ylabel('convergence time')

    fig_dir4.tight_layout()
    plt.savefig('lineselect-dir4.png')

    #combined
    fig_combined, ax1 = plt.subplots(figsize=(5,2.5))
    ax1.bar(bar_c, result['combined'][0], 0.6, yerr=[err for err in result['combined'][1]], capsize=4, color='forestgreen')
    ax1.set_xlabel('number of lines')
    ax1.set_ylabel('best_accuracy')
    ax1.set_ylim([0.6,1.1])
    plt.xticks(bar_c, range(16,6,-1))

    ax2 = ax1.twinx()
    ax2.plot(bar_c, result['combined'][2], linewidth=4, color='darkorange')
    ax2.set_ylabel('convergence time')

    fig_combined.tight_layout()
    plt.savefig('lineselect-combined.png')

def plot_tfusion(result, accuracy):
    plt.figure(figsize=(6,4))
    bar_c = np.array([1.25, 3.75, 6.25, 8.75])
    offset = 0
    for i, subset in enumerate(repro):
        plt.bar(bar_c-0.8+offset, result[subset][2], 0.4, label=label[i])
        offset += 0.4

    plt.legend(loc='upper right')
    plt.ylim([0,1000])
    plt.xlabel('fusion methods')
    plt.ylabel('convergence time')
    plt.xticks(bar_c, method_name)
    plt.savefig('tfusion-convergence time.png')

    plt.figure(figsize=(6,4))
    offset = 0
    for i, subset in enumerate(repro):
        plt.bar(bar_c-0.8+offset, result[subset][0], 0.4, yerr=[err for err in result[subset][1]], capsize=3, label=label[i])
        offset += 0.4
    plt.legend(loc='lower right')
    plt.ylim([0.6,1.1])
    plt.xticks(bar_c, method_name)
    plt.xlabel('fusion methods')
    plt.ylabel('best accuracy')
    plt.savefig('tfusion-accuracy.png')

    #curve comparison
    plt.figure(figsize=(6,2.1))
    for i, method in enumerate(method_list):
        plt.plot(np.arange(1,1001,30), [accuracy[i][epoch] for epoch in range(1000) if epoch%30==0], label=method_name[i], linewidth=3)
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.savefig('testcurve.png')

def plot_refusion(result, accuracy):
    plt.figure(figsize=(6,4))
    bar_c = np.array([1.25, 3.75, 6.25, 8.75])
    #offset = 0
    for i, feature in enumerate(label_con):
        plt.bar(bar_c, result[feature][2], 0.4, label=label_con[i])
        #offset += 0.4

    plt.legend(loc='upper right')
    plt.ylim([0,1000])
    plt.xlabel('fusion methods')
    plt.ylabel('convergence time')
    plt.xticks(bar_c, method_name)
    plt.savefig('reduced-tfusion-convergence time.png')

    plt.figure(figsize=(6,4))
    #offset = 0
    for i, feature in enumerate(label_con):
        plt.bar(bar_c, result[feature][0], 0.4, yerr=[err for err in result[feature][1]], capsize=3, label=label_con[i])
        #offset += 0.4
    plt.legend(loc='lower right')
    plt.ylim([0.6,1.1])
    plt.xticks(bar_c, method_name)
    plt.xlabel('fusion methods')
    plt.ylabel('best accuracy')
    plt.savefig('reduced-tfusion-accuracy.png')
    plt.show()

    ##curve comparison
    #plt.figure(figsize=(6,2.1))
    #for i, method in enumerate(method_list):
    #    plt.plot(np.arange(1,1001,30), [accuracy[i][epoch] for epoch in range(1000) if epoch%30==0], label=method_name[i], linewidth=3)
    #plt.legend()
    #plt.ylabel('accuracy')
    #plt.xlabel('epochs')
    #plt.savefig('reduced-testcurve.png')
    #plt.show()

def plot_partline(result):
    plt.figure(figsize=(6,4))
    bar_c = np.arange(0.5,6.5,1)
    plt.bar(bar_c, result['dir2'][0], 0.5, yerr=[err for err in result['dir2'][1]], capsize=3)

    plt.ylim([0.6,1.1])
    plt.xticks(bar_c, part_line)
    plt.xlabel('line combination')
    plt.ylabel('best accuracy')
    plt.savefig('part_line_best_accuracy.png')

if __name__ == '__main__':
    
    #load and evaluation for feature selection
    split = 'feature-selection'
    for name in feature:
        for subset in repro:
            #print(name, subset)
            accu, var, conv, _ = eval_record(name, subset, split, 1000)
            result[subset][0].append(accu)
            result[subset][1].append(var)
            result[subset][2].append(conv)
    #print(result)
    plot_feature(result)

    #load and evaluation for line selection
    result = {'dir1':[[],[],[]], 'dir2':[[],[],[]], 'dir3':[[],[],[]], 'dir4':[[],[],[]], 'combined':[[],[],[]]}
    split = 'line-selection'
    for line in range(16,6,-1):
        for subset in repro:
            #print(name, subset)
            accu, var, conv, _ = eval_record(line, subset, split, 1000)
            result[subset][0].append(accu)
            result[subset][1].append(var)
            result[subset][2].append(conv)
    #print(result)
    plot_line(result)
    result = {'dir1':[[],[],[]], 'dir2':[[],[],[]], 'dir3':[[],[],[]], 'dir4':[[],[],[]], 'combined':[[],[],[]]}
    for group in part_line:
        accu, var, conv, _ = eval_record(group, 'dir2', split, 1000)
        result['dir2'][0].append(accu)
        result['dir2'][1].append(var)
        result['dir2'][2].append(conv)
    plot_partline(result)

    #load and evaluation for temporal fusion
    result = {'dir1':[[],[],[]], 'dir2':[[],[],[]], 'dir3':[[],[],[]], 'dir4':[[],[],[]], 'combined':[[],[],[]]}
    split = 't-fusion'
    accuracy = []
    for method in method_list:
        for subset in repro:
            accu, var, conv, accu_record = eval_record(method, subset, split, 1000)
            result[subset][0].append(accu)
            result[subset][1].append(var)
            result[subset][2].append(conv)
            if subset == 'dir2':
                accuracy.append(accu_record)
    #print(result)
    plot_tfusion(result, accuracy)

    ##load and evaluation for reduced temporal fusion
    #result = {'fix':[[],[],[]], 'random':[[],[],[]]}
    #split = 'reduce-t-fusion'
    #accuracy = []
    #for method in method_list:
    #    for feature in label_con:
    #        accu, var, conv, accu_record = eval_con_record(feature, method, 'dir2', split, 1000)
    #        result[feature][0].append(accu)
    #        result[feature][1].append(var)
    #        result[feature][2].append(conv)
    #        accuracy.append(accu_record)
    ##print(result)
    #plot_refusion(result, accuracy)

    #data volume reduction
    result = {'ins':[[],[]], 'xyz':[[],[]], 'depth':[[],[]], 'fix':[[],[]], 'random':[[],[]]}
    split = 'data-reduce'
    for volume in volume_list:
        for feature in label_con:
            accu, var, _, _ = eval_record(feature, volume, split, 1000)
            result[feature][0].append(accu)
            result[feature][1].append(var)
    for feature in label_con:
        print(result[feature][1])
