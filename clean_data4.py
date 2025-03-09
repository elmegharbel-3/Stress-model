import csv
import pandas as pd
import os

main_folder = "Data_set"  # Change this to your actual folder path
output_folder = "cleaned"  # Folder to save cleaned files
stress_mapping = {'Standing':0,
    'Left Fall while Standing':1,
    'Right Fall while Standing':1,
    'Forward Fall while Standing':1 , 
    'Sitting':0,
    'Left Fall while Sitting':1,
    'Right Fall while Sitting':1,
    'Forward Fall while Sitting':1,
    'Walking':0,
    'Relax':0,
    'Anxiety':1,
    'Rest':0,
    'Sad':1,
    'Motivate':0,
    'Funny':0,
    'Stress Ball':0,
    'Hand at Rest':0,
    'Fist':1
}

os.makedirs(output_folder, exist_ok=True)
c = 1
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    subfiles = os.listdir(subfolder_path)
    print(subfiles)
    PID = os.path.join(subfolder_path,subfiles[0])
    PID_BPM = os.path.join(subfolder_path,subfiles[1])
    df1 = pd.read_csv(PID_BPM)
    df2 = pd.read_csv(PID)
    print("file number",c)

    df1 = pd.read_csv("Data_set/PID7/PID_7_BPM.csv")
    df2 = pd.read_csv("Data_set/PID7/PID_7.csv")

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    df1 = df1[["Datetime","hr","spo2","features"]]
    df2 = df2[["Datetime","GSR(kohm)","Features"]]


    df1["Datetime"] = df1["Datetime"].str.strip()
    df2["Datetime"] = df2["Datetime"].str.strip()

    df1['Datetime'] = df1['Datetime'].astype(str)
    df2['Datetime'] = df2['Datetime'].astype(str)

    df1.dropna(subset=["Datetime"], inplace=True)
    df2.dropna(subset=["Datetime"], inplace=True)


    merged_df = df1.merge(df2, on='Datetime')

    merged_df['features'] = merged_df['features'].replace(stress_mapping)

    print(merged_df['features'].unique())
    merged_df = merged_df.rename(columns={'features':'stress_state'})

    output_file = os.path.join(output_folder, f"cleaned_{c}.csv")
    merged_df.to_csv(output_file, index=False)
    c += 1





        