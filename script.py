# %%
# !pip install langchain-chroma
# !pip install chromadb
# !pip install open-clip-torch
# !pip install torch
# !pip install datasets
# !pip install sentence_transformers
# !pip install -q kaggle
# !pip install opencv-python


# %%
# downloading dataset from kaggle

# from kaggle import KaggleApi

# # create Kaggle API object
# kaggle_api = KaggleApi()
# kaggle_api.authenticate()

# # download dataset
# kaggle_api.dataset_download_files('vikashrajluhaniwal/fashion-images', path='./', unzip=True)


# %%
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()

# %%
from chromadb.utils.data_loaders import ImageLoader
data_loader = ImageLoader()

# %%
import chromadb

client = chromadb.Client()

try:
  collections = client.list_collections()
  for collection in collections:
      client.delete_collection(collection.name)
except Exception:
  pass

collection = client.create_collection(
    name='image_search',
    embedding_function=embedding_function,
    data_loader=data_loader)

# %%
import os
import cv2

basepath = 'data/all/'

images_path = os.listdir(basepath)

print("total :", len(images_path))

ids, images_list = [], []
# print(images_path)
for idx,image in enumerate(images_path):
  # print(idx)
  name = f"{basepath}/{image}"
  numpy_image = cv2.imread(name)

  ids.append(str(image))
  images_list.append(numpy_image)

# print(ids)
collection.add(
  ids=ids,
  images=images_list
)


