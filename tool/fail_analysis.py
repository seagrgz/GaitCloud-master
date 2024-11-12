#!/usr/bin/env python
import yaml
import os

def main():
    timestamp = ['[2024-10-26_09:35:21.579475]', '[2024-10-25_10:21:30.734829]']
    results = load_results(timestamp)
    records = {}
    for i,epoch in enumerate(results.keys()):
        if i == 0:
            for attr in results[epoch].keys():
                records[attr] = {}
                for var in results[epoch][attr].keys():
                    records[attr][var] = {}
                    for sample in results[epoch][attr][var]['failed_name'].keys():
                        records[attr][var][sample] = [1, epoch[:2]+epoch[7:-5]]
        else:
            for attr in results[epoch].keys():
                for var in results[epoch][attr].keys():
                    for sample in results[epoch][attr][var]['failed_name'].keys():
                        if sample in records[attr][var].keys():
                            records[attr][var][sample][0] += 1
                            records[attr][var][sample].append(epoch[:2]+epoch[7:-5])
                        else:
                            records[attr][var][sample] = [1, epoch[:2]+epoch[7:-5]]

    for attr in records.keys():
        for var in records[attr].keys():
            for sample, fail_count in records[attr][var].items():
                if fail_count[0] == len(results.keys()):
                    print(sample)
    if len(timestamp) > 1:
        with open('fail_analysis.yaml', 'w') as f:
            yaml.dump(records, f, default_flow_style=False)
    else:
        with open('{}/fail_analysis/results.yaml'.format(timestamp[0]), 'w') as f:
            yaml.dump(records, f, default_flow_style=False)
    f.close()

def load_results(stamps):
    results = {}
    for i, time in enumerate(stamps):
        root = '{}/fail_analysis'.format(time)
        os.system('rm -f {}/results.yaml'.format(root))
        filenames = sorted(os.listdir(root), key=lambda file: int(file[5:-5]))
        for f in filenames:
            with open(os.path.join(root, f), 'r') as F:
                results['{}_'.format(i)+f] = yaml.safe_load(F)
            F.close()
    return results

if __name__ == '__main__':
    main()
