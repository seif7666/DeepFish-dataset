import pandas as pd
import re
import os
from PIL import Image


class EstimationLoader:
    def __init__(self, path, dataset_path: str) -> None:
        self.__dataset = pd.read_csv(path)
        self.__DATASET_PATH = dataset_path
        # self.__rename()
        self.__filter()
        print(self.__dataset.columns)

    def __filter(self):
        for i in range(len(self.__dataset)):
            row = self.__dataset.iloc[i]
            if not self.__is_file_exists(row["file"] + ".jpg"):
                self.__dataset.loc[i, "file"] = None

        length = len(self.__dataset)
        self.__dataset = self.__dataset.dropna()
        print(f"{length- len(self.__dataset)} rows were removed from image dataset!")
        print(len(self.__dataset))

    def __is_file_exists(self, path):
        path = self.__DATASET_PATH + path
        return os.path.exists(path)

    def __rename(self):
        del self.__dataset["Unnamed: 0"]
        for i in range(len(self.__dataset)):
            val = self.__dataset.loc[i, "file"]
            vals = val.split("-")
            code = re.findall("[A-Z]+", vals[1])[0]
            number = re.split("[A-Z]+", vals[1])[1]
            val = vals[0] + "-" + code + "." + number + ".jpg"
            self.__dataset.loc[i, "file"] = val

    def __getitem__(self, idx):
        row = self.__dataset.iloc[idx]
        image = self.__loadImage(row["file"] + ".jpg")
        return {"image": image, "gt_bbox": row["bbox"], "size": row["size (cm)"]}

    def __len__(self):
        return len(self.__dataset)

    def __str__(self) -> str:
        return str(self.__dataset.columns)

    def __loadImage(self, path):
        return Image.open(self.__DATASET_PATH + path)


if __name__ == "__main__":
    load = EstimationLoader("../size_estimation_homography_DeepFish.csv", "../DATASET/")
    print(load[0])
