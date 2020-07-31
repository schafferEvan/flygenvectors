#!/usr/bin/python
"""
Master script to run regression & plotting on all datasets
"""

import make_regression_plots_single_fly as mk_reg_plt


run_exp_list = [
            ['2018_08_24','fly3_run1'],
            ['2018_08_24','fly2_run2'],
            ['2019_07_01','fly2'],
            ['2019_10_14','fly3'],
            ['2019_06_28','fly2'],
            ['2019_10_14','fly2'],
            ['2019_10_18','fly3'],
            ['2019_10_21','fly1'],
            ['2019_10_10','fly3'],
            ['2019_08_14','fly1']]


feed_exp_list = [
            ['2019_04_18','fly2'],
            ['2019_04_22','fly1'],
            ['2019_04_22','fly3'],
            ['2019_04_24','fly3'],
            ['2019_04_24','fly1'],
            ['2019_04_25','fly3'],
            ['2019_05_07','fly1'],
            ['2019_03_12','fly4'],
            ['2019_02_19','fly1'],
            ['2019_02_26','fly1_2']]



main_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/dataShare/_main/' #'/Volumes/data1/_flygenvectors_dataShare/_main/_sparseLines/'
main_fig_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/figs/' #'/Volumes/data1/figsAndMovies/figures/'
remake_pickle = True   # rerun regression
activity = 'dFF'        # metric of neural activity {'dFF', 'rate'}, the latter requires deconvolution
split_behav = False     # treat behavior from each trial as separate regressor
elasticNet = False      # run regression with elastic net regularization (alternative is OLS)

input_dict = {
    'main_dir':main_dir, 'main_fig_dir':main_fig_dir, 'exp_date':None, 'fly_num':None, 
    'remake_pickle':remake_pickle, 'activity':activity, 'split_behav':split_behav, 'elasticNet':elasticNet
    }


exp_list = run_exp_list
for i in range(len(exp_list)):
    print('\n\n\n ----- '+exp_list[i][0]+' '+exp_list[i][1]+' -----\n')
    input_dict['exp_date'] = exp_list[i][0]
    input_dict['fly_num'] = exp_list[i][1]
    mk_reg_plt.run_all(input_dict=input_dict)

