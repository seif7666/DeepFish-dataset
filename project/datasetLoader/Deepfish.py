from torch.utils.data import Dataset
from JsonReader import createJsonReader, getDictFromJsonReader
from ImageLoader import ImageLoader
from AnnotationsLoader import AnnotationsLoader


class DeepFishDataset(Dataset):
    def __init__(self, dataset_path: str, json_path: str) -> None:
        super().__init__()
        createJsonReader(json_path)
        self.__image_loader = ImageLoader(getDictFromJsonReader("images"), dataset_path)
        self.__annotation_loader = AnnotationsLoader(
            getDictFromJsonReader("annotations"), self.__image_loader
        )

    def __getitem__(self, index):
        image_row = self.__image_loader[index]
        annotations = self.__annotation_loader[image_row["id"]]
        image = self.__image_loader.load_image(image_row["file_name"])
        return {"image": image, "gt_annots": annotations}


if __name__ == "__main__":
    dataset = DeepFishDataset("../DATASET/", "../coco_format_fish_data.json")
    print(dataset[0])
