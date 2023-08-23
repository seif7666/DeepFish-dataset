import pandas as pd
from loader import Loader
from ImageLoader import ImageLoader
from JsonReader import createJsonReader,getDictFromJsonReader

class AnnotationsLoader(Loader):
    def __init__(self, dictionary, imageLoader:ImageLoader) -> None:
        super().__init__(dictionary)
        del self._dataset['segmentation']
        self.__filter(imageLoader)

    def __filter(self,imageLoader:ImageLoader):
        for i in range(len(self)):
            row= super().__getitem__(i)
            if imageLoader[row['image_id']] is None:
                self._dataset.loc[i,'image_id']= None
        length= len(self)
        self._dataset= self._dataset.dropna()
        print(f'{length- len(self)} rows were removed from annotations dataset!')

    def __getitem__(self, idx) -> object:
        """
        Takes Image ID and returns all annotations corresponding to this image.
        """
        return self._dataset.loc[self._dataset['image_id']==idx]


if __name__=='__main__':
    createJsonReader('../coco_format_fish_data.json')
    annDictionary= getDictFromJsonReader('annotations')
    imgDictionary= getDictFromJsonReader('images')
    loader= ImageLoader(imgDictionary,'../DATASET/')
    annotationLoader= AnnotationsLoader(annDictionary,loader)
    print(len(annotationLoader))

    