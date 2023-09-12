from retinanet.PipelineModel import PipelineModel
import torch
from torchvision.transforms import transforms
from datasetLoader.dataloader import load_image
import numpy as np
import cv2
import pandas as pd


def draw_caption(image, p1,p2, caption):
    b=np.array([p1[0],p1[1],p2[0],p2[1]],dtype=int)
    print(b)
    cv2.putText(image, str(caption), (((b[0]+b[0])//2), (2*b[1]+b[3])//3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, str(caption), (((b[0]+b[0])//2), (2*b[1]+b[3])//3), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

class ModelProxy:
    DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self) -> None:
        self.__model= None
        self.__set_model()
        self.__original_image=None
        self.__error_message=''
        self.__model_predictions=None
    def __set_model(self):
        try:
            torch.set_grad_enabled(False)
            state_dict= torch.load('..\\bestModel.pt',torch.device(ModelProxy.DEVICE))
            self.__model= PipelineModel(13)
            self.__model.to(ModelProxy.DEVICE)
            self.__model.load_state_dict(state_dict)
            self.__model.eval()
        except Exception as e:
            self.__error_message= f'Failed initiating the model\n\n{e}'
            self.get_error_message()
            raise Exception

    def load_image(self,image_path)->bool:
        try:
            input_tensor,self.__original_image= load_image(image_path)
            input_tensor: torch.Tensor= input_tensor
            print(input_tensor.sum())
            self.__original_image= self.__original_image.numpy().astype(np.uint8)
            scores, classification, transformed_anchors =\
                            self.__model(torch.unsqueeze(input_tensor.to(ModelProxy.DEVICE),dim=0).float())
            transformed_anchors=transformed_anchors.numpy()
            predictions= pd.DataFrame({
               'scores': scores.numpy(),
               'class': classification.numpy(),
               'X1':transformed_anchors[:,0],
               'Y1':transformed_anchors[:,1],
               'X2':transformed_anchors[:,2],
               'Y2':transformed_anchors[:,3],
               'size':transformed_anchors[:,4] 
            })
            predictions=predictions.loc[predictions['class']==5]
            predictions.drop('class',axis=1,inplace=True)
            predictions.reset_index(inplace=True)
            predictions.drop('index',axis=1,inplace=True)
            predictions['scores']=predictions['scores'].round(2)
            columns=predictions.columns[1:]
            predictions[columns]= predictions[columns].astype(int)
            self.__model_predictions=predictions
            print(self.__model_predictions)
            return True
        except Exception as e:
            e.with_traceback()
            self.__error_message=e
            self.get_error_message()
            return False
    def display_img(self):
        cv2.imshow('Original', cv2.cvtColor(self.__original_image,cv2.COLOR_BGR2RGB))
        pass
    def get_scores(self)->np.ndarray:
        return self.__model_predictions['scores'].to_numpy()
    
    def get_image(self,chosen_idxs:list):
        chosen= self.__model_predictions.iloc[chosen_idxs]
        image= self.__original_image.copy()
        for pred in range(len(chosen)):
            row= chosen.iloc[pred]
            point1= (int(row['X1']),int(row['Y1']))
            point2= (int(row['X2']),int(row['Y2']))
            size=row['size']
            self.__draw_on_image(image,point1,point2,size)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    def __draw_on_image(self,image,point1,point2,size):
        draw_caption(image, point1,point2, size)
        cv2.rectangle(image, point1, point2, color=(0,0,255), thickness=2)

    def get_error_message(self)->str:
        print(self.__error_message)

if __name__=='__main__':
    model= ModelProxy()
    model.load_image('D:\Personal\Torpedo\Deepfish\Project\DATASET\\29_04_21-B21.jpg')
    print(model.get_scores())
    cv2.imshow('BOX',model.get_image([10]))
    # model.display_img()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



