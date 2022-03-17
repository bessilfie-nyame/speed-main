import pandas as pd 
import numpy as np
import os
import shutil
import csv

root_path = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/'
target_root = '/srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/bespeed/speed-main/data'

train_path = os.path.join(root_path, 'drive360challenge_train.csv')
val_path = os.path.join(root_path, 'drive360challenge_validation.csv')

train_out = os.path.join(target_root, 'train/dataset/rgb')
val_out = os.path.join(target_root, 'validation/dataset/rgb')

train_label = os.path.join(target_root, 'train_target.csv')
val_label = os.path.join(target_root, 'val_target.csv')

train_annotations = os.path.join(target_root, 'train/dataset/annotations.txt')
val_annotations = os.path.join(target_root, 'validation/dataset/annotations.txt')

MAX_SPEED = 36.893229

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

def avg_speed(infile, normalize=False):
    
    for episode in genbatches(infile):
        speed_list = None
        new_list = []
        for i, (_, speed, _) in enumerate(episode[:10*(len(episode)//10)]):
            current_speed = speed
            if normalize:
                current_speed /= MAX_SPEED
            new_list.append(current_speed)

            if len(new_list) == 10:
                speed_list = new_list
                new_list = []
                yield np.mean(speed_list)

def write_avg_speed(inpath, file_name, normalize=True):
    with open(file_name, mode='w') as csvfile:
        fieldname = ['avgCanSpeed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldname)

        writer.writeheader()

        for speed in avg_speed(inpath, normalize=normalize):
            writer.writerow({'avgCanSpeed': speed})
                    
def create_annotations(infile, file_name, normalize=False):
    with open(file_name, "w") as file:
        n = 1
        for speed in avg_speed(infile, normalize=normalize):
            file.writelines(f"rgb/video_{n}  1  10  {speed} \n")
            n += 1


if __name__ == '__main__':
    # batch(train_path, train_out)
    # batch(val_path, val_out)

    # write_avg_speed(train_path, train_label, normalize=False)
    # write_avg_speed(val_path, val_label, normalize=False)

    create_annotations(train_path, train_annotations, normalize=False)
    create_annotations(val_path, val_annotations, normalize=False)
    
  
       



