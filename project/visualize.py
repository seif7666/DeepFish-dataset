import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasetLoader.dataloader import AspectRatioBasedSampler,EstimatedDeepFish,transforms,Resizer,Normalizer,UnNormalizer,Permuter
from retinanet.PipelineModel import PipelineModel

def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def evaluate(model:torch.nn.Module,data:dict,unnormalize:UnNormalizer,classes:np.ndarray):
    # model.cuda()
    scores, classification, transformed_anchors = model(torch.unsqueeze(data['img'].cuda(),dim=0).float())
    print('Evaluation complete!')
    idxs = np.where(scores.cpu()>0.5)
    img = np.array(255 * unnormalize(data['img'][:, :, :])).copy()
    img[img<0] = 0
    img[img>255] = 255
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    print(transformed_anchors)
    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        # print(scores[idxs[0][j]])
        label_name = classes[int(classification[idxs[0][j]])]+" "+str(scores[idxs[0][j].item()])
        draw_caption(img, (x1, y1, x2, y2), label_name)

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # print(label_name)

    number= data['number']
    labels= data['annot'][:number,4].to(torch.int)
    print(labels)
    for i in labels:
          print(classes[i])
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



               
def main():
    dataset=EstimatedDeepFish('./Project/size_estimation_homography_DeepFish.csv','./Project/DATASET/',transforms.Compose([Normalizer(), Resizer(480,480),Permuter()]))
    model= PipelineModel(len(dataset.num_classes()[0]))
    model_params= torch.load('./model_final11.pt')
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


if __name__=='__main__':
    main()
    pass