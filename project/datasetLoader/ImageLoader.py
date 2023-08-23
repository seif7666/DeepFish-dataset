import pandas as pd
import os
from loader import Loader
from JsonReader import createJsonReader,getDictFromJsonReader
from PIL import Image
import torchvision.transforms as tr
class ImageLoader(Loader):

    def __init__(self,dictionary, dataset_path) -> None:
        super().__init__(dictionary)
        self.__dataset_path= dataset_path
        self.__filter()
        self.__transform= tr.Compose([
            tr.PILToTensor(),
            tr.Resize((1024,1024)),
        ])

    def __filter(self):
        for i in range(len(self)):
            row= super().__getitem__(i)
            if not self.__is_file_exists(row['file_name']):
                self._dataset.loc[i,'file_name']= None
        length= len(self)
        self._dataset= self._dataset.dropna()
        print(f'{length- len(self)} rows were removed from image dataset!')

    def __is_file_exists(self,path):
        path= self.__dataset_path+path
        return os.path.exists(path)
    
    def getByID(self,id) ->object:
        element= self._dataset.loc[self._dataset['id']==id]
        if len(element):
            return element.iloc[0]
        return None 

    def load_image(self,path):
        return self.__transform(Image.open(self.__dataset_path+path))
    
if __name__=='__main__':
    createJsonReader('../coco_format_fish_data.json')
    dictionary= getDictFromJsonReader('images')
    loader= ImageLoader(dictionary,'../DATASET/')



