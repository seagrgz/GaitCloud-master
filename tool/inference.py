#!/usr/bin/env python
import os
import sys
import yaml
import time
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.append('./')
sys.path.append('util')
from util import config
from util.gait_database import GaitDataset
import models

def get_parser():
    parser = argparse.ArgumentParser(description='Load model parameters for inference')
    parser.add_argument('--time', type=str, help='training timestamp')
    parser.add_argument('--visual', type=bool, default=False, help='if visualization')
    timestamp = parser.parse_args().time
    visual = parser.parse_args().visual
    cfg = config.load_cfg_from_cfg_file('[{}]/config.yaml'.format(timestamp))
    return cfg, timestamp, visual

def main():
    args, timestamp, visual = get_parser()
    print('config loaded')
    data_root = args.data_root+'/test'
    data_list = [item[:-4] for item in sorted(os.listdir(data_root))]
    target = list(dict.fromkeys([name[-4:] for name in data_list]))
    args.target = np.arange(250)
    args.timestamp = timestamp

    #model initialization
    Collate_fn = None
    args.symbol = None
    device = torch.device('cuda')
    args.dtype = torch.float

    print('loading {} in {}'.format(args.structure, timestamp))
    Model = getattr(models, args.structure)
    model = Model(args, args.feature)
    model.load_state_dict(torch.load('[{}]/best_view.pth'.format(timestamp)))
    model.eval().to(device)
    if os.path.exists('[{}]/best_var.pth'.format(args.timestamp)):
        model_var = Model(args, args.feature)
        model_var.load_state_dict(torch.load('[{}]/best_var.pth'.format(timestamp)))
        model_var.eval().to(device)
    else:
        model_var = None
    print('network loaded')

    Evaluator = getattr(models, 'MetricEvaluator')
    accuracy_calculator = Evaluator()

    mean_view = []
    mean_var = []
    #for noise in [0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1]:
    for noise in [0]:
        print('set noise to {}'.format(noise))
        if visual:
            data_root = args.data_root
            trainv_set = GaitDataset(split='inference', data_root=os.path.join(data_root,'train'), args=args, datalist=args.visual_train_names)
            trainv_loader = torch.utils.data.DataLoader(trainv_set, batch_size=8, num_workers=args.workers, collate_fn=Collate_fn, drop_last=False)
            testv_set = GaitDataset(split='inference', data_root=os.path.join(data_root,'test'), args=args, datalist=args.visual_test_names)
            testv_loader = torch.utils.data.DataLoader(testv_set, batch_size=8, num_workers=args.workers, collate_fn=Collate_fn, drop_last=False)
            #train visual
            for batch_idx, (data, labels, metainfo) in enumerate(trainv_loader):
                data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with autocast(dtype=torch.bfloat16):
                    _, visual_embed = model(data, labels, training=False, positions=positions, symbol=args.symbol, visual=True)
                for name in metainfo[1]:
                    visual_save(args.timestamp, 'best', name, visual_embed, metainfo[1].index(name))
            #test visual
            for batch_idx, (data, labels, metainfo) in enumerate(testv_loader):
                data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with autocast(dtype=torch.bfloat16):
                    _, visual_embed = model(data, labels, training=False, positions=positions, symbol=args.symbol, visual=True)
                for name in metainfo[1]:
                    visual_save(args.timestamp, 'best', name, visual_embed, metainfo[1].index(name))
        else:
            test_set = GaitDataset(split='inference', data_root=data_root.format(noise), args=args, datalist=data_list)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, num_workers=args.workers, collate_fn=Collate_fn, drop_last=False)
            test_embeddings = []
            test_labels = []
            test_targets = []

            for batch_idx, (data, labels, metainfo) in enumerate(test_loader):
                if batch_idx%20 == 0:
                    print('{} sample calculated'.format(batch_idx*256))
                data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        embeddings, _ = model(data, labels, training=False, positions=positions, symbol=args.symbol)
                test_embeddings.append(embeddings.detach().to('cpu'))
                test_labels.append(labels.detach().to('cpu'))
                test_targets += metainfo[1]
            test_embeddings = torch.cat(test_embeddings)
            test_labels = torch.cat(test_labels)
            print('embeddings calculation complete')

            #view
            view_accuracy = []
            view_dist = []
            for gallery in args.splits_view:
                gallery_targets = [item for item in test_targets if gallery in item]
                gallery_embeddings = test_embeddings[[test_targets.index(item) for item in gallery_targets]]
                gallery_labels = test_labels[[test_targets.index(item) for item in gallery_targets]]
                for probe in args.splits_view:
                    if not gallery == probe:
                        probe_targets = [item for item in test_targets if probe in item]
                        probe_embeddings = test_embeddings[[test_targets.index(item) for item in probe_targets]]
                        probe_labels = test_labels[[test_targets.index(item) for item in probe_targets]]
                        accuracy, _ = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
                        view_accuracy.append(accuracy)
                        view_dist.append(len(probe_targets))
            mean_view.append(sum([view_accuracy[i]*view_dist[i] for i in range(len(view_dist))])/sum(view_dist))
            print('view evaluation complete')

            #variance
            var_accuracy = []
            var_dist = []
            if not model_var == None: #best_variance model is not best_view
                for batch_idx, (data, labels, metainfo) in enumerate(test_loader):
                    if batch_idx%20 == 0:
                        print('{} sample calculated'.format(batch_idx*256))
                    data, labels, positions = data.to(device).to(args.dtype), labels.to(device).to(args.dtype), metainfo[0].to(device).to(args.dtype)
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16):
                            embeddings, _ = model_var(data, labels, training=False, positions=positions, symbol=args.symbol)
                    test_embeddings.append(embeddings.detach().to('cpu'))
                    test_labels.append(labels.detach().to('cpu'))
                    test_targets += metainfo[1]

                test_embeddings = torch.cat(test_embeddings)
                test_labels = torch.cat(test_labels)

            gallery_targets = [item for item in test_targets if '00-nm' in item]
            gallery_embeddings = test_embeddings[[test_targets.index(item) for item in gallery_targets]]
            gallery_labels = test_labels[[test_targets.index(item) for item in gallery_targets]]
            fail_results = {}
            fail_results_view = {}
            for probe in args.splits_variance:
                if not probe == '00-nm':
                    probe_targets = [item for item in test_targets if probe in item]
                    probe_embeddings = test_embeddings[[test_targets.index(item) for item in probe_targets]]
                    probe_labels = test_labels[[test_targets.index(item) for item in probe_targets]]
                    accuracy, failed_info = accuracy_calculator.rank_1_accuracy(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels)
                    fail_results[probe], fail_results_view[probe] = fail_analyze(failed_info, probe_targets, gallery_targets)
                    var_accuracy.append(accuracy)
                    var_dist.append(len(probe_targets))
            mean_var.append(sum([var_accuracy[i]*var_dist[i] for i in range(len(var_dist))])/sum(var_dist))
            print('variance evaluation complete')
    if not visual:
        print('mean_view\n', mean_view)
        print('mean_var\n', mean_var)

        with open('[{}]/fail_analysis.yaml'.format(args.timestamp), 'w') as f:
            yaml.dump(fail_results, f, allow_unicode=True, default_flow_style=False)
        f.close()
        with open('[{}]/fail_analysis_view.yaml'.format(args.timestamp), 'w') as f:
            yaml.dump(fail_results_view, f, allow_unicode=True, default_flow_style=False)
        f.close()
        #np.save('{}/robustness.npy'.format(args.timestamp), np.array([mean_var, mean_view]))

def visual_save(timestamp, epoch, name, embeds, idx):
    print('record {}'.format(name))
    if not os.path.exists('[{}]/{}'.format(timestamp, name)):
        os.system('mkdir [{}]/{}'.format(timestamp, name))
    for item in embeds.keys():
        #print(len(visual_embed[item]), len(metainfo[1]))
        np.save('[{}]/{}/{}_{}.npy'.format(timestamp, name, epoch, item), embeds[item][idx])

def fail_analyze(failed_info, probe_names, gallery_names):
    results = {}
    results_view = {}
    failed_names = [probe_names[i] for i in failed_info[0]]
    failed_preds = [gallery_names[i] for i in failed_info[1]]
    for name in probe_names:
        var, view, _ = name.split('_')
        if var in results.keys():
            if name in failed_names:
                results[var]['failed_name'][name] = failed_preds[failed_names.index(name)]
            results[var]['count'] += 1
        else:
            results[var] = {}
            if name in failed_names:
                results[var]['failed_name'] = {name:failed_preds[failed_names.index(name)]}
            else:
                results[var]['failed_name'] = {}
            results[var]['count'] = 1

        if view in results_view.keys():
            if name in failed_names:
                results_view[view]['failed_name'][name] = failed_preds[failed_names.index(name)]
            results_view[view]['count'] += 1
        else:
            results_view[view] = {}
            if name in failed_names:
                results_view[view]['failed_name'] = {name:failed_preds[failed_names.index(name)]}
            else:
                results_view[view]['failed_name'] = {}
            results_view[view]['count'] = 1

    for var in results.keys():
        results[var]['accuracy'] = (results[var]['count'] - len(results[var]['failed_name']))/results[var]['count']
    for view in results_view.keys():
        results_view[view]['accuracy'] = (results_view[view]['count'] - len(results_view[view]['failed_name']))/results_view[view]['count']
    return results, results_view

def draw_distribution(results, results_view, timestamp):
    for attr in results.keys():
        #variance
        x_labels = [name for name in results[attr].keys()]
        x_locs = np.arange(0, 5.9, 6/(len(x_labels)+1))[1:]
        width = 6/(len(x_labels)+1)/3
        accuracy = [results[attr][name]['accuracy'] for name in x_labels]
        count = [results[attr][name]['count'] for name in x_labels]
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        ax.bar(x_locs-width/2, accuracy, width, label='Accuracy', color='tab:blue', zorder=2)
        addlabels(x_locs, width, accuracy)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0,1)
        ax.set_xticks(x_locs, x_labels, fontsize=4)
        ax.set_zorder(1)
        ax.set_frame_on(False)
        ax_c = ax.twinx()
        ax_c.bar(x_locs+width/2, count, width, label='Count', color='tab:orange', zorder=2)
        ax_c.set_ylabel('Count')
        fig.legend()
        plt.savefig('[{}]/analysis/{}.png'.format(timestamp, attr))
        plt.close()

        #view
        x_labels = [name for name in results_view[attr].keys()]
        x_locs = np.arange(0, 5.9, 6/(len(x_labels)+1))[1:]
        width = 6/(len(x_labels)+1)/3
        accuracy = [results_view[attr][name]['accuracy'] for name in x_labels]
        count = [results_view[attr][name]['count'] for name in x_labels]
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        ax.bar(x_locs-width/2, accuracy, width, label='Accuracy', color='tab:blue')
        ax.set_xticks(x_locs, x_labels, fontsize=5)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0,1)
        ax_c = ax.twinx()
        ax_c.bar(x_locs+width/2, count, width, label='Count', color='tab:orange')
        ax_c.set_ylabel('Count')
        fig.legend()
        plt.savefig('[{}]/analysis/{}_view.png'.format(timestamp, attr))
        plt.close()

def addlabels(locs, width, y):
    for i in range(len(locs)):
        plt.text(locs[i]-width/2, y[i], round(y[i], 4), ha='center', fontsize=8, zorder=10)

def draw_robustness(var_accu, view_accu, timestamp):
    fig = plt.figure(figsize = (6,4),dpi=300) 
    noise = np.arange(0,0.105,0.005)
    noise = (3**0.5)*noise
    plt.plot(noise, np.asarray(var_accu), label = 'variance')
    plt.plot(noise, np.asarray(view_accu), label = 'view')
    fig.legend()
    plt.savefig('[{}]/robustness.png'.format(timestamp))
    plt.close()

if __name__ == '__main__':
    main()

    #timestamp = '[2024-10-11_07:43:50.842896]'
    #with open('{}/fail_analysis.yaml'.format(timestamp), 'r') as f:
    #    results = yaml.safe_load(f)
    #f.close()
    #with open('{}/fail_analysis_view.yaml'.format(timestamp), 'r') as f:
    #    results_view = yaml.safe_load(f)
    #f.close()
    #draw_distribution(results, results_view, timestamp)
