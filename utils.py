import pandas as pd 
import numpy as np
import os
import shutil

root_path = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/'

train_path = os.path.join(root_path, 'drive360challenge_train.csv')
val_path = os.path.join(root_path, 'drive360challenge_validation.csv')

train_out = '/srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/bespeed/speed-main/data/train/dataset/rgb'
val_out = '/srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/bespeed/speed-main/data/validation/dataset/rgb'

def genbatches(file):
    df = pd.read_csv(file)
    speed_df = df[["cameraFront", "canSpeed", "chapter"]]

    for i in speed_df['chapter'].unique():
        yield speed_df[speed_df.chapter == i].values.tolist()


def batch(infile, outfile):
    speed_gen = genbatches(infile)   
    batch = 1
    vid_num = 1  

    for episode in speed_gen:
        sub_dir = os.path.join(outfile, f"video_{vid_num}") 

        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        for i, (im_path, _, _) in enumerate(episode[:10*(len(episode)//10)]):
            
            
            src = os.path.join(root_path, im_path)

            if batch > 10:
                batch = 1
                vid_num += 1
                sub_dir = os.path.join(outfile, f"video_{vid_num}")

                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)

            if batch < 10:
                fname = f"img_0000{batch}.jpg"
            else:
                fname = f"img_000{batch}.jpg"

            dst = os.path.join(sub_dir, fname)
            shutil.copyfile(src, dst)

            batch += 1

def avg_speed(infile, outfile):
    
    for episode in genbatches(infile):
        speed_list = None
        new_list = []
        for i, (_, speed, _) in enumerate(episode[:10*(len(episode)//10)]):
            new_list.append(speed)

            if len(new_list) == 10:
                speed_list = new_list
                new_list = []
                yield np.mean(speed_list)
                    

if __name__ == '__main__':
    # batch(train_path, train_out)
    batch(val_path, val_out)
  
       



