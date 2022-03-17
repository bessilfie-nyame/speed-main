import pandas as pd

train_path = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/drive360challenge_train.csv'
val_path = '/srv/beegfs02/scratch/aegis_guardian/data/Datasets_ML_Pipeline/Drive360Challenge/Drive360Images/drive360challenge_validation.csv'

def meta(file, head=True, max_speed=False, avg_speed=False, columns=None, num_chapters=False):
    """
        Get relevant meta data from csv file
    """
    
    df = pd.read_csv(file)
    if head:
        print("First Five Rows: ")
        print(df.head(10))
        print(" ")
        print("All Columns: ")
        print(df.columns)

    if columns is not None:
        print(" ")
        print("Interested Columns: ")
        print(df[columns].head())
        print(" ")
        print("cameraFront - paths to the images recorded by the front camera")
        print("canSpeed - ground truth speed of the vehicle")
        print("chapter - recording episodes")

    if max_speed:
        print(" ")
        print(f"Maximum Speed: {df['canSpeed'].max()}")

    if avg_speed:
        print(" ")
        print(f"avg speed first n: {df['canSpeed'].head(10)}")
        print(f"avg speed first n: {df['canSpeed'].head(10).mean()}")

    if num_chapters:
        print(" ")
        print(f"Number of episodes: {df['chapter'].nunique()}")
        print(" ")

    return f"Total Number of Entries: {len(df)}"
        

print("Train Dataset... ")
print(" ")
print(meta(train_path, max_speed=True, avg_speed=True))

# print(" ")
# print(" ")

# print("Validation Dataset... ")
# print(" ")
# print(meta(val_path, max_speed=True, columns=["cameraFront", "canSpeed", "chapter"], num_chapters=True))

# def genbatches(file):
#     df = pd.read_csv(file)
#     speed_df = df[["cameraFront", "canSpeed", "chapter"]]

#     for i in range(572):
#         yield speed_df[speed_df.chapter == i]

# print("Showing some episodes from the Dataset")
# print("======================================")
# print(" ")

# print("Train Dataset... ")
# print(" ")
# tspeed_gen = genbatches(train_path)  
# t_all = 0     
# for s in tspeed_gen:
#     print(f"Episode: {t_all}")
#     print(s)
#     t_all += 1
#     if t_all == 5+1:
#         break

# print(" ")
# print(" ")
# print("Validation Dataset... ")
# print(" ")
# vspeed_gen = genbatches(val_path)  
# v_all = 0     
# for k in vspeed_gen:
#     print(f"Episode: {v_all}")
#     print(k)
#     v_all += 1
#     if v_all == 5+1:
#         break