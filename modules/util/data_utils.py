import os
import zipfile
import pandas as pd
from kaggle import KaggleApi


class Fer2013():
    DIR = 'datasets/'

    # Initializing variables
    all_data = []
    train_data = []
    test_data = []


    def download(self):

        dataset = 'deadskull7/fer2013'
        path = 'datasets'

        api = KaggleApi()
        api.authenticate()

        if not os.path.exists(path+"/fer2013.zip"):
            api.dataset_download_files(dataset, path)

        if not os.path.exists(path+"/fer2013.csv"):
            with zipfile.ZipFile(path+"/fer2013.zip", "r") as z:
                z.extractall(path)

        if not os.path.exists(path+"/raw/train_all.csv"):
            os.mkdir(path+"/build")
            os.mkdir(path+"/raw")
            os.mkdir(path+"/processed")
            df = pd.read_csv(path+"/fer2013.csv")
            df[df["Usage"] == "Training"].to_csv(path+"/raw/train_all.csv")
            df[df["Usage"] == "PublicTest"].to_csv(path+"/raw/val_all.csv")
            df[df["Usage"] == "PrivateTest"].to_csv(path+"/raw/test_all.csv")

