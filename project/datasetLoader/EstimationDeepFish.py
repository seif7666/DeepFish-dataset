from torch.utils.data import Dataset,DataLoader
try:
    from .EstimationLoader import EstimationLoader
except:
    from EstimationLoader import EstimationLoader

from torchvision.transforms import Compose,ToTensor,Resize
class EstimatedDeepFish(Dataset):
    def __init__(self, csv_path:str, dataset_path:str) -> None:
        super().__init__()
        self.__estimationLoader= EstimationLoader(csv_path,dataset_path)
        self.__transform= Compose([ToTensor(),Resize((1024,1024))])
    def __getitem__(self, index) -> dict:
        data= self.__estimationLoader[index]
        data['image']= self.__transform(data['image'])
        return data
    def __len__(self):
        return len(self.__estimationLoader)
    
if __name__=='__main__':
    dataset= EstimatedDeepFish('size_estimation_homography_DeepFish.csv', 'DATASET\\')
    print(dataset[0])
    dataloader= DataLoader(dataset,16)
    print(next(iter(dataloader)))

    