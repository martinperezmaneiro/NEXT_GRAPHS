import os
import glob
import itertools


pressure = '13bar'
action = 'train'
basedir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/{}/'.format(pressure)

train_data = 'dataset_{}_graph_nn_all.pt'.format(pressure)
file_in = basedir + train_data

train_folder_structure = 'neural_network/lr_{lr}_do_{do}_bs_{bs}_nconv_{nconv}/'
train_dir = basedir + train_folder_structure
prediction_file = train_dir + 'test_prediction.h5'

configTemp_path = '/home/usc/ie/mpm/NEXT_graphs/scripts/explore_train_params/'
taskTemp = '/home/usc/ie/mpm/NEXT_graphs/templates/taskTemplate.sh'
script_dir = '/home/usc/ie/mpm/NEXT_graphs/scripts/main.py'

lrs = [1e-4, 1e-3]
dos = [0.5, 0.1]
bss = [32, 64, 128]
nconvs = [2, 3, 4]

combinations = list(itertools.product(lrs, dos, bss, nconvs))

if __name__ == "__main__":

    print('Creating tasks...')

    #open the template to use it
    task_file = open(taskTemp).read()

    for comb in combinations:
        lr, do, bs, nconv = comb

        #the task will contain various comands of the type:
        #python create_label_dataset.py  conf_n.conf

        ######## CREATE CONFIGS (needed one for each task)#########
        #these are the files to write in the config file, the input and the output
        train_path  = train_dir.format(lr = lr, do = do, bs = bs, nconv = nconv)
        checkpoint_dir  = train_path + 'checkpoint_dir/'
        tensorboard_dir = train_path + 'tensorboard_dir/'
        if action == 'train':
            os.makedirs(checkpoint_dir)
            os.makedirs(tensorboard_dir)

            #open the template to use it
            config_file = open(configTemp_path + 'GCNClassTemplate.conf').read()
            #create the config file to write the template on it
            config = train_path + "config_train.conf"
            with open(config, "w") as config_write:
        
                config_write.write(config_file.format(file_in = file_in, 
                                                    nconv = nconv, 
                                                    dropout = do, 
                                                    lr = lr, 
                                                    bs = bs, 
                                                    check_dir = checkpoint_dir, 
                                                    tb_dir = tensorboard_dir))
        if action == 'predict':
            #search for the last checkpoint
            checkpoints = sorted(glob.glob(checkpoint_dir + '*.pth.tar'), key = lambda x: int(x.split('_')[-1].split('.')[0]))
            #open the template to use it
            config_file = open(configTemp_path + 'GCNClassTemplate_pred.conf').read()
            #create the config file to write the template on it
            config = train_path + "config_pred.conf"
            with open(config, "w") as config_write:
                config_write.write(config_file.format(file_in = file_in, 
                                                    nconv = nconv, 
                                                    dropout = do, 
                                                    bs = bs, 
                                                    out_file = prediction_file, 
                                                    saved_weights = checkpoints[-1]))

        #we create the commands to be written in the job file, such as the scripth path
        commands = "python {script_dir} -a {action} -conf {config_directory}".format(script_dir = script_dir,
                                                                                     action = action,
                                                                                     config_directory = config)

        ######### CREATE TASKS (each one with one labelling command)############
        #create the task file to write the template on it
        taskname = "task_train.sh" if action == 'train' else "task_pred.sh"
        task = train_path + taskname
        with open(task, "w") as task_write:
            task_write.write(task_file.format(commands = commands))
        os.chmod(task, 0o744) #I think this gives executable permises

