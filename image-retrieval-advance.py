import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from tqdm import tqdm

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
HNSW_SPACE = "hnsw:space"


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path


embedding_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=np.array(image))
    return embedding


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data-query), axis=axis_batch_size)


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query) ** 2, axis=axis_batch_size)


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def get_score(root_img_path, query_path, size, method_cal_distance):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)  # array of image and their path
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = method_cal_distance(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
    return files_path


def plot_results(image_path, files_path, results, method_name):
    query_image = Image.open(image_path).resize((448, 448))
    images = [query_image]
    class_name = []
    for id_img in results['ids'][0]:
        id_img = int(id_img.split('_')[-1])
        img_path = files_path[id_img]
        img = Image.open(img_path).resize((448, 448))
        images.append(img)
        class_name.append(img_path.split('/')[2])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(method_name, fontsize=16)

    # Iterate through images and plot them
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if i == 0:
            ax.set_title(f"Query Image: {image_path.split('/')[2]}")
        else:
            ax.set_title(f"Top {i+1}: {class_name[i-1]}")
        ax.axis('off')  # Hide axes
    # Display the plot
    plt.show()


data_path = f'{ROOT}/train'
files_path = get_files_path(path=data_path)


def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    try:
        collection.add(embeddings=embeddings, ids=ids)
    except:
        print("Fail adding data")


# Create a Chroma Client
chroma_client = chromadb.Client()
# Create a collection
l2_collection = chroma_client.get_or_create_collection(
    name="l2_collection", metadata={HNSW_SPACE: "l2"})
add_embedding(collection=l2_collection, files_path=files_path)

# Create a collection
cosine_collection = chroma_client.get_or_create_collection(
    name="Cosine_collection", metadata={HNSW_SPACE: "cosine"})
add_embedding(collection=cosine_collection, files_path=files_path)


def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(
        query_embeddings=[query_embedding], n_results=n_results)
    return results


img = Image.open(f'{ROOT}/train/African_crocodile/n01697457_260.JPEG')
get_single_image_embedding(image=img)

test_path = f"{ROOT}/test"
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
l2_results = search(image_path=test_path,
                    collection=l2_collection, n_results=5)
plot_results(image_path=test_path, files_path=files_path,
             results=l2_results, method_name="l2")

cosine_results = search(image_path=test_path,
                        collection=cosine_collection, n_results=5)
plot_results(image_path=test_path, files_path=files_path,
             results=cosine_results, method_name="Cosine Similarity")
