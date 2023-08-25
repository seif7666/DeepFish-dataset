import json
import pandas as pd


class JsonReader:
    def __init__(self, json_path) -> None:
        file = open(json_path)
        self.__dictionaries = json.load(file)
        print(self.__dictionaries.keys())

    def get(self, dictionaryName: str):
        return self.__dictionaries[dictionaryName]


jsonReader: JsonReader = None


def createJsonReader(path):
    global jsonReader
    if jsonReader is None:
        jsonReader = JsonReader(path)


def getDictFromJsonReader(name):
    return jsonReader.get(name)
