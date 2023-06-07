import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_df(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    values = ea.Tags()['scalars']
    
    if not values:
        #p = path.split('/')
        #warnings.warn("The file {}, in the folder {}, does not contain any data".format(p[-1], p[-3]))
        return pd.DataFrame()
    else:
        #take one of them that share wall time and step to be the base
        df = pd.DataFrame(ea.Scalars(values[0]))[['wall_time', 'step']]

        #iterate over the values to get the data
        for val in values:
            colname = val.replace('/', '_')
            df = df.assign(**{colname : pd.DataFrame(ea.Scalars(val)).value})
        
        df['wall_time'] = df['wall_time'].diff().fillna(0) / 60
        return df
    

    

    
