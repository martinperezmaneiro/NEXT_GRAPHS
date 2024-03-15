import os
import glob
from production_variables import *

in_filename_structure  = "{dir}/" + in_filename
out_filename_structure = "{dir}/" + out_filename

if __name__ == "__main__":

    print('Creating tasks...')

    #take the template path
    task_temp = taskTemp_dir + taskTemp_filename
    #open the template to use it
    task_file = open(task_temp).read()

    for f in files_in:
        n = get_file_number(f)
        commands = ""

        #the task will contain various comands of the type:
        #python create_label_dataset.py  conf_n.conf

        ######## CREATE CONFIGS (needed one for each task)#########
        #these are the files to write in the config file, the input and the output
        file_in  = in_filename_structure.format(dir = indir,
                                                num = n, 
                                                tag = tag)
        file_out = out_filename_structure.format(dir = proddir,
                                                 num = n, 
                                                 tag = tag)
        #take the template path
        config_temp = configTemp_dir + configTemp_filename
        #open the template to use it
        config_file = open(config_temp).read()
        #create the config file to write the template on it
        config = confdir + "conf_{n}.conf".format(n = n)
        with open(config, "w") as config_write:
            config_write.write(config_file.format(file_in = file_in, 
                                                  fileout = file_out))

        #we create the commands to be written in the job file, such as the scripth path
        commands = "python {script_directory} {config_directory}".format(script_directory = scriptdir,
                                                                         config_directory = config)

        ######### CREATE TASKS (each one with one labelling command)############
        #create the task file to write the template on it
        task = taskdir + "task_{n}.sh".format(n = n)
        with open(task, "w") as task_write:
            task_write.write(task_file.format(commands = commands))
        os.chmod(task, 0o744) #I think this gives executable permises
    print(f"{len(files_in)} tasks created")
