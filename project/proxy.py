from retinanet.PipelineModel import PipelineModel
import torch
from torchvision.transforms import transforms
from datasetLoader.dataloader import load_image
import numpy as np
class ModelProxy:
    DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self) -> None:
        self.__set_model()
        self.__original_image=None
        self.__error_message=''
    def __set_model(self):
        try:
            state_dict= torch.load('../models/bestModel.pt',torch.device(ModelProxy.DEVICE))
            self.__model= PipelineModel(13)
            self.__model.to(ModelProxy.DEVICE)
            self.__model.load_state_dict(state_dict)
            self.__model.eval()
        except Exception as e:
            self.__error_message= f'Failed initiating the model\n\n{e}'
            self.get_error_message()

    def load_image(self,image_path)->bool:
        try:
            input_tensor,self.__original_image= load_image(image_path)
            input_tensor: torch.Tensor= input_tensor
            scores, classification, transformed_anchors =\
                            self.__model(torch.unsqueeze(input_tensor.to(ModelProxy.DEVICE),dim=0).float())
            self.pagrus_classes= classification==5
            print(self.pagrus_classes)
        except Exception as e:
            e.with_traceback()
            self.__error_message=e
            self.get_error_message()
            return False
    
    def display_img(self):
        pass
    def get_scores(self)->dict:
        pass
    def get_image(self,chosen_scores:list):
        pass

    def get_error_message(self)->str:
        print(self.__error_message)

if __name__=='__main__':
    model= ModelProxy()
    model.load_image('~/Work/Torpedo/Torpedo24/Deepfish/DeepFish-dataset/Project/DATASET/08_04_21-B.15.jpg')



