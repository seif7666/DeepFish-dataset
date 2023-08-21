import pandas as pd


class Loader:

    def __init__(self, dictionary) -> None:
        self._dataset= pd.DataFrame(dictionary)    
    def __str__(self) -> str:
        return str(self._dataset.head())
    def __getitem__(self, idx) -> object:
        return self._dataset.iloc[idx]
    def __len__(self):
        return len(self._dataset)
        