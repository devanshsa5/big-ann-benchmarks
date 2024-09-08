import argparse
import os
import numpy as np

from benchmark.datasets import DATASETS
from benchmark.streaming.load_runbook import load_runbook

def get_range_start_end(entry):
    return np.arange(entry['start'],  entry['end'], dtype=np.uint32)

def get_next_set(ids: np.ndarray, entry):
    operation = entry['operation']
    if operation == 'insert':
        range = get_range_start_end(entry)
        return np.union1d(ids, range)
    elif operation == 'delete':
        range = get_range_start_end(entry)
        return np.setdiff1d(ids, range, assume_unique=True)
    elif operation == 'search':
        return ids
    else:
        raise ValueError('Undefined entry in runbook')
        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)

def output_gt(ds, ids, step, gt_cmdline, runbook_path):
    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[ids]

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, 'step') + str(step) 
    os.makedirs(dir, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(ids.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        ids.tofile(tf)
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little')) #npts
        f.write(ds.d.to_bytes(4, byteorder='little'))
        data_slice.tofile(f)
    
    gt_cmdline += ' --base_file ' + data_file 
    gt_cmdline += ' --gt_file ' + gt_file
    gt_cmdline += ' --tags_file ' + tags_file
    print("Executing cmdline: ", gt_cmdline)
    os.system(gt_cmdline)
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path'
    )
    parser.add_argument(
        '--private_query',
        action='store_true'
    )
    parser.add_argument(
        '--gt_cmdline_tool',
        required=True
    )
    parser.add_argument(
        '--download',
        action='store_true'
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    max_pts, runbook = load_runbook(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    common_cmd = args.gt_cmdline_tool + ' --dist_fn ' 
    distance = ds.distance()
    if distance == 'euclidean':
        common_cmd += 'l2'
    elif distance == 'ip':
        common_cmd += 'mips'
    else:
        raise RuntimeError('Invalid metric')

    common_cmd += ' --data_type '
    dtype = ds.dtype
    if dtype == 'float32':
        common_cmd += 'float'
    elif dtype == 'int8':
        common_cmd += 'int8'
    elif dtype == 'uint8':
        common_cmd += 'uint8'
    else:
        raise RuntimeError('Invalid datatype')
        
    common_cmd += ' --K 100'
    common_cmd += ' --query_file ' + os.path.join(ds.basedir, query_file)

    step = 1
    ids = np.empty(0, dtype=np.uint32)
    for entry in runbook:
        if step == 1:
            ids = get_range_start_end(entry)
        else:
            ids = get_next_set(ids, entry)
        print(ids)
        if entry['operation'] == 'search':
            output_gt(ds, ids, step, common_cmd, args.runbook_file)
        step += 1

if __name__ == '__main__':
    main()