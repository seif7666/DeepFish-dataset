import pandas as pd
import os
from loader import Loader
from JsonReader import createJsonReader,getDictFromJsonReader
class ImageLoader(Loader):

    def __init__(self,dictionary, dataset_path) -> None:
        super().__init__(dictionary)
        self.__dataset_path= dataset_path
        self.__filter()

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
    
    def __getitem__(self, idx) -> object:
        element= self._dataset.loc[self._dataset['id']==idx]
        if len(element):
            return element.iloc[0]
        return None 
    
if __name__=='__main__':
    createJsonReader('../coco_format_fish_data.json')
    dictionary= getDictFromJsonReader('images')
    loader= ImageLoader(dictionary,'../DATASET/')



