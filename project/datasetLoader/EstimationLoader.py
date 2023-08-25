import pandas as pd
import re
import os
from PIL import Image
import numpy as np
class EstimationLoader:
    def __init__(self, path, dataset_path:str) -> None:
        self.__dataset= pd.read_csv(path)
        self.__DATASET_PATH= dataset_path
        self.__image_files=self.__dataset['file'].unique().tolist()

        # self.__rename()
        self.__filter()
        print(self.__dataset.columns)
        
    def __filter(self):
        self.__dataset.drop(self.__dataset[self.__dataset['class'] != 'Pagrus pagrus'].index, inplace=True)
        self.__image_files=self.__dataset['file'].unique().tolist()

        deleted_files= []
        for i in range(len(self.__image_files)):
            file= self.__image_files[i]
            if (not self.__is_file_exists(file+'.jpg')) and (not self.__is_file_exists(file+'.JPG')):
                deleted_files.append(file) 

        length= len(self.__image_files)
        for file in deleted_files:
            self.__image_files.remove(file)
        
        print(f'{length- len(self.__image_files)} rows were removed from image dataset!')
        for file in deleted_files:
            self.__dataset.drop(self.__dataset[self.__dataset.file==file].index,inplace=True)
        print(len(self.__dataset))
    def __is_file_exists(self,path):
        path= self.__DATASET_PATH+path
        return os.path.exists(path)        
    
    def __rename(self):
        del self.__dataset['Unnamed: 0']
        for  i in range(len(self.__dataset)):
            val=self.__dataset.loc[i,'file']
            vals= val.split('-')
            code=re.findall('[A-Z]+',vals[1])[0]
            number= re.split('[A-Z]+',vals[1])[1]
            val= vals[0]+'-'+code+'.'+number+'.jpg'
            self.__dataset.loc[i,'file']= val

    def __getitem__(self,idx):
        filename= self.__image_files[idx]
        image= self.__loadImage(filename+'.jpg')
        annotations= self.__dataset.loc[self.__dataset['file']==filename]
        bboxes= np.zeros((13,4))
        sizes= np.zeros((13,1))
        for i in range(len(annotations)):
            string= annotations['bbox'].iloc[i]
            string= string.split(']')[0].split('[')[1]
            nums= string.split(',')
            for c in range(4):
                bboxes[i,c]=float(nums[c])
            sizes[i,0]= annotations['size (cm)'].iloc[i]
        
        return {
            'image': image,
            'gt_bbox': bboxes,
            'size': sizes,  
            'number': len(annotations)        
        }
    def __len__(self):
        return len(self.__image_files)
    def __str__(self) -> str:
        return str(self.__dataset.columns)
    def __loadImage(self,path):
        return Image.open(self.__DATASET_PATH+path)
if __name__ == '__main__':
    load= EstimationLoader('size_estimation_homography_DeepFish.csv','DATASET/')
    print(load[0])
    