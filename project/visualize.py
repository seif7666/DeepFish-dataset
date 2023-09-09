import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasetLoader.dataloader import AspectRatioBasedSampler,EstimatedDeepFish,transforms,Resizer,Normalizer,UnNormalizer,Permuter
from retinanet.PipelineModel import PipelineModel

def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (((b[0]+b[0])//2), (2*b[1]+b[3])//3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (((b[0]+b[0])//2), (2*b[1]+b[3])//3), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def evaluate(model:torch.nn.Module,data:dict,unnormalize:UnNormalizer,classes:np.ndarray):
    # model.cuda()
    scores, classification, transformed_anchors = model(torch.unsqueeze(data['img'].cuda(),dim=0).float())
    print('Evaluation complete!')
    idxs = np.where(scores.cpu()>0.1)
    print(scores)
    img = np.array(255 * unnormalize(data['img'][:, :, :])).copy()
    img[img<0] = 0
    img[img>255] = 255
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    # cv2.imshow('Before',img)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred_img,target_img= img.copy(),img.copy()
    
    # print(transformed_anchors)
    for j in range(idxs[0].shape[0]):
        # print(classification[idxs[0][j]] )
        if classification[idxs[0][j]] != 5:
              continue
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        print(f'Length is {bbox[4]}')
        # print(scores[idxs[0][j]].item())
        # label_name = classes[int(classification[idxs[0][j]])]+" "+str(scores[idxs[0][j]].item())

        label_name = str(int(bbox[4].item()))
        # print(label_name)
        draw_caption(pred_img, (x1, y1, x2, y2), label_name)
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), color=(0,100+j*10 , 255//(j+1)), thickness=2)
        # print(label_name)
    for j in range(data['number']):
        bbox = data['annot'][j]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        # print(f'Length is {bbox[4]}')
        # print(scores[idxs[0][j]].item())
        label_name = str(int(bbox[5].item()))
        print(label_name)
        draw_caption(target_img, (x1, y1, x2, y2), label_name)
        cv2.rectangle(target_img, (x1, y1), (x2, y2), color=(0,100+j*10 , 255//(j+1)), thickness=2)
        
          
    number= data['number']
    lengths= data['annot'][:number,5].to(torch.int)
    print(lengths)
    # for i in lengths:
        #   print(classes[i])
    cv2.imshow('Predictions', pred_img)
    cv2.imshow('Target', target_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



               
def main():
    dataset=EstimatedDeepFish('../Project/size_estimation_homography_DeepFish.csv','../Project/DATASET/',transforms.Compose([Normalizer(), Resizer(480,480),Permuter()]),False)
    model= PipelineModel(13)
    model_params= torch.load('../models/bestModel2.pt')
    model.load_state_dict(model_params)
    model.cuda()
    print('Model is loaded!')
    # dataloader= DataLoader(dataset,batch_size=4,num_workers=1)
    model.eval()
    unnormalize= UnNormalizer()
    while True:
            try:
                index= int(input('Enter an index: ')) % len(dataset)
                data= dataset[index]
                print('Data loaded successfully')
                evaluate(model,data,unnormalize,dataset.num_classes()[0])
            except Exception as e:
                   e.with_traceback()
                   print(e)
                   break

def test():
    dataset=EstimatedDeepFish('../Project/size_estimation_homography_DeepFish.csv','../Project/DATASET/',transforms.Compose([Normalizer(), Resizer(480,480),Permuter()]),False)
    dataloader= DataLoader(dataset,1)
    data= next(iter(dataloader))
    data['img'],data['annot']=data['img'].cuda(),data['annot'].cuda()
    model= PipelineModel(13)
    model_params= torch.load('../bestModel.pt')
    model.load_state_dict(model_params)
    model.train()
    model.cuda()
    model(data)

      
if __name__=='__main__':
    main()
    # test()
    pass