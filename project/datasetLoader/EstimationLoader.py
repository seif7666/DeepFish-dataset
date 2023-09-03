import pandas as pd
import re
import os
from PIL import Image
import numpy as np
import torch
import sklearn.preprocessing as pre


class EstimationLoader:
    def __init__(self, path, dataset_path: str) -> None:
        self.__dataset = pd.read_csv(path)
        self.__DATASET_PATH = dataset_path
        self.__image_files = self.__dataset["file"].unique().tolist()
        self.__encoder = pre.OrdinalEncoder()
        uniques = np.array(self.__dataset["class"].unique()).reshape(-1, 1)
        # print()
        self.__encoder.fit(uniques)
        # print(self.__encoder.transform(np.array([['Pagrus pagrus']])))
        # self.__rename()
        self.__filter()
        del self.__dataset["Unnamed: 0"]
        print(self.__dataset.columns)

    def getNumClasses(self):
        return self.__encoder.categories_

    def __filter(self):
        # self.__dataset.drop(self.__dataset[self.__dataset['class'] != 'Pagrus pagrus'].index, inplace=True)
        self.__image_files = self.__dataset["file"].unique().tolist()

        deleted_files = []
        for i in range(len(self.__image_files)):
            file = self.__image_files[i]
            if (not self.__is_file_exists(file + ".jpg")) and (
                not self.__is_file_exists(file + ".JPG")
            ):
                deleted_files.append(file)

        length = len(self.__image_files)
        for file in deleted_files:
            self.__image_files.remove(file)

        print(
            f"{length- len(self.__image_files)} rows were removed from image dataset!"
        )
        for file in deleted_files:
            self.__dataset.drop(
                self.__dataset[self.__dataset.file == file].index, inplace=True
            )
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
        from time import time

        first_time = time()
        filename = self.__image_files[idx]
        second_time = time()
        print(f"Loading {filename} took {second_time - first_time} seconds")
        try:
            image = self.__loadImage(filename + ".jpg")
        except:
            image = self.__loadImage(filename + ".JPG")

        annotations = self.__dataset.loc[self.__dataset["file"] == filename]
        third_time = time()
        print(f"Loading annotations took {third_time - second_time} seconds")
        length = 32  # len(annotations)
        bboxes = torch.zeros((length, 5))
        sizes = torch.zeros((length, 1))
        for i in range(len(annotations)):
            string = annotations["bbox"].iloc[i]
            string = string.split("]")[0].split("[")[1]
            nums = string.split(",")
            for c in range(4):
                bboxes[i, c] = float(nums[c])
            sizes[i, 0] = annotations["size (cm)"].iloc[i]
        fourth_time = time()
        print(f"Loading bboxes took {fourth_time - third_time} seconds")
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        # print(annotations['class'])
        classes = self.__encoder.transform(
            annotations["class"].to_numpy().reshape((-1, 1))
        )
        classes = torch.from_numpy(classes.flatten())
        bboxes[: len(annotations), 4] = classes
        fifth_time = time()
        print(f"Loading classes took {fifth_time - fourth_time} seconds")

        return {
            "image": image,
            "annots": bboxes,
            "size": sizes,
            "number": len(annotations),
        }

    def __len__(self):
        return len(self.__image_files)

    def __str__(self) -> str:
        return str(self.__dataset.columns)

    def __loadImage(self, path):
        path = self.__DATASET_PATH + path
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0


if __name__ == "__main__":
    load = EstimationLoader(
        "Project/size_estimation_homography_DeepFish.csv", "Project/DATASET/"
    )
    for i in range(10):
        print(load[i]["annots"][:, 4])
