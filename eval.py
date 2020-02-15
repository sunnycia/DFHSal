import os, glob, sys, time, shutil
import numpy as np
import config as cfg

save_base = os.path.join(cfg.metric_base, cfg.test_dataset_name)
if not os.path.isdir(save_base):
    os.makedirs(save_base)

_,sal_base, fixa_dir,dens_dir = cfg.get_input_output_dir(cfg.test_dataset_name)
# print(sal_base,dens_dir,fixa_dir) get_input_output_dir

sal_subdir_list =  [ name for name in os.listdir(sal_base) if os.path.isdir(os.path.join(sal_base, name)) ]

for sal_subdir in sal_subdir_list:
    sal_dir = os.path.join(sal_base, sal_subdir)
    var_str = 'save_base=\'%s\';dsname=\'%s\';sal_dir=\'%s\';dens_dir=\'%s\';fixa_dir=\'%s\';other_num=%s;'% (save_base, cfg.test_dataset_name, sal_dir, dens_dir, fixa_dir,str(cfg.other_num))
    if cfg.no_sauc!=0:
        print('no sauc calc')
        var_str = var_str+'no_sauc=\'1\';';
    cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric_code\');%s metric_image_base;' % var_str
    if cfg.metric_debug == 0:
        cmd = cmd + 'exit()"'
    else:
        cmd = cmd + '"'

    print('running:', cmd)
    if os.path.isfile(os.path.join(save_base, cfg.test_dataset_name+'_'+sal_subdir+'.mat')):
        continue
    else:
        # print (os.path.join(save_base, cfg.test_dataset_name+'_'+sal_subdir+'.mat'))
        os.system(cmd)
    if cfg.metric_debug == 1:
        exit()
    # finish_list.append(sal_subdir)

# stastics
if cfg.statistics_flag:
    cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric_code\');save_base=\'%s\';metric_stastics;exit()"' % save_base
    os.system(cmd)

## write to latex

if cfg.latex_file == True:
    #read matlab file
    import scipy.io as sio
    # save_base
    save_base = os.path.join(cfg.metric_base, cfg.test_dataset_name)
    # metric-name index list
    MI_list=[('CC', 0),('SIM', 1),('AUC', 2),('KLD', 5),('NSS', 6)]
    model_file_list = os.listdir(save_base)
    wf = open(cfg.output_file, 'a') ## append mode
    first_line_flag = 1
    model_number = len(model_file_list)
    for model_file in model_file_list:
        metric_path = os.path.join(save_base, model_file)
        model_name = model_file.split('.')[0].split('_')[-1]
        model_name = model_name.upper()
        metric_structure = sio.loadmat(metric_path)
        saliency_score = metric_structure['saliency_score']
        line = ''
        if first_line_flag:
            line+=('\\multirow{{{}}}{{*}}{{{}}}'.format(model_number, os.path.basename(save_base).replace('_', '\_')))
            first_line_flag = 0
        line+=('& {}'.format(model_name))

        for metric_name, index in MI_list:
            metric_avg = np.around(np.mean(saliency_score[index]), 4)
            # metric_std = ...
            line+=('& %.4f'%metric_avg)
        wf.write(line+'\\\\\n')
    wf.write('\\midrule\n')
    wf.write('\\midrule\n')
    wf.close()
        