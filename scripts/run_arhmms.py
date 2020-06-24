import os
import subprocess
import argparse
import shutil
import json
import commentjson

from behavenet import get_user_dir

user_dir = '/home/mattw/'
code_dir = '/home/mattw/Dropbox/github/behavenet'
grid_search_file = os.path.join(code_dir, 'behavenet/fitting/arhmm_grid_search.py')
config_files = {
    'data': os.path.join(user_dir, '.behavenet/fly_run_params.json'),
    'model': os.path.join(user_dir, '.behavenet/arhmm_labels_model_fly.json'),
    'training': os.path.join(user_dir, '.behavenet/arhmm_training_fly.json'),
    'compute': os.path.join(user_dir, '.behavenet/arhmm_compute_fly.json')
}
KAPPAS = [1e4, 1e6, 1e8]  # kappas to use for sticky transitions


def run_main(args):

    # make a copy of model config
    dirname = os.path.dirname(config_files['model'])
    filename = os.path.basename(config_files['model']).split('.')[0]
    tmp_file = os.path.join(dirname, filename + '_tmp.json')
    shutil.copy(config_files['model'], tmp_file)
    config_files['model'] = tmp_file
           
    # get list of transitions
    transitions = []
    if args.stationary:
        transitions.append('stationary')
    if args.sticky:
        transitions.append('sticky')
    if args.recurrent:
        transitions.append('recurrent')
    if args.recurrent_only:
        transitions.append('recurrent_only')

    # loop over transitions
    for transition in transitions:

        # modify configs
        if transition == 'sticky':
            kappas = KAPPAS
        else:
            kappas = 0
            
        update_config(config_files['model'], 'kappa', kappas)
        update_config(config_files['model'], 'transitions', transition)

        call_str = [
            'python',
            grid_search_file,
            '--data_config', config_files['data'],
            '--model_config', config_files['model'],
            '--training_config', config_files['training'],
            '--compute_config', config_files['compute']
        ]
        subprocess.call(' '.join(call_str), shell=True)

    # remove copy of model config
    os.remove(config_files['model'])


def update_config(file, key, value):

    # load json file as dict
    config = commentjson.load(open(file, 'r'))

    # update value
    config[key] = value

    # resave file
    with open(file, 'w') as f:
        json.dump(config, f, sort_keys=False, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--stationary', action='store_true', default=False)
    parser.add_argument('--sticky', action='store_true', default=False)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--recurrent_only', action='store_true', default=False)
    namespace, _ = parser.parse_known_args()
    run_main(namespace)

