#!/usr/bin/env python
import yaml
import os

def main():
    timestamp = '[2024-10-25_10:21:30.734829]'
    results = load_results(timestamp)
    records = {}
    for i,epoch in enumerate(results.keys()):
        if i == 0:
            for attr in results[epoch].keys():
                records[attr] = {}
                for var in results[epoch][attr].keys():
                    records[attr][var] = {}
                    for sample in results[epoch][attr][var]['failed_name'].keys():
                        records[attr][var][sample] = [1, epoch[5:-5]]
        else:
            for attr in results[epoch].keys():
                for var in results[epoch][attr].keys():
                    for sample in results[epoch][attr][var]['failed_name'].keys():
                        if sample in records[attr][var].keys():
                            records[attr][var][sample][0] += 1
                            records[attr][var][sample].append(epoch[5:-5])
                        else:
                            records[attr][var][sample] = [1, epoch[5:-5]]

    for attr in records.keys():
        for var in records[attr].keys():
            for sample, fail_count in records[attr][var].items():
                if fail_count[0] == len(results.keys()):
                    print(sample)
    with open('{}/fail_analysis/results.yaml'.format(timestamp), 'w') as f:
        yaml.dump(records, f, default_flow_style=False)
    f.close()

def load_results(stamp):
    root = '{}/fail_analysis'.format(stamp)
    filenames = [f for f in sorted(os.listdir(root)) if not f == 'results.yaml']
    results = {}
    for f in filenames:
        with open(os.path.join(root, f), 'r') as F:
            results[f] = yaml.safe_load(F)
        F.close()
    return results

if __name__ == '__main__':
    main()
