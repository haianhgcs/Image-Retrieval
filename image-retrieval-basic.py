import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))


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


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data-query), axis=axis_batch_size)


print("L1:", absolute_difference(np.array([1, 2, 3]), np.array([4, 5, 6])))


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query) ** 2, axis=axis_batch_size)


print("L2:", mean_square_difference(np.array([1, 2, 3]), np.array([4, 5, 6])))


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


print("Cosine:", cosine_similarity(np.array([1, 0, 1]), np.array([1, 1, 0])))


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


print("Pearson:", correlation_coefficient(np.array([2, 4]), np.array([1, 2])))


def get_score(root_img_path, query_path, size, method_cal_distance):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)  # array of image and their path
            rates = method_cal_distance(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def plot_results(query_path, ls_path_score, reverse, method_name):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    fig.suptitle(method_name, fontsize=16)
    plt.imshow(read_image_from_path(query_path, size=(448, 448)))
    plt.title(f"Query Image: {query_path.split('/')[2]}", fontsize=16)
    plt.axis("off")
    for i, path in enumerate(sorted(ls_path_score, key=lambda x: x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448, 448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()


root_img_path = f"{ROOT}/train/"
size = (448, 448)

query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"

query, ls_path_score = get_score(
    root_img_path, query_path, size, absolute_difference)
plot_results(query_path, ls_path_score, reverse=False,
             method_name="absolute difference")

query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
query, ls_path_score = get_score(
    root_img_path, query_path, size, absolute_difference)
plot_results(query_path, ls_path_score, reverse=False,
             method_name="absolute difference")

query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
query, ls_path_score = get_score(
    root_img_path, query_path, size, mean_square_difference)
plot_results(query_path, ls_path_score, reverse=False,
             method_name="mean square difference")

query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
query, ls_path_score = get_score(
    root_img_path, query_path, size, mean_square_difference)
plot_results(query_path, ls_path_score, reverse=False,
             method_name="mean square difference")

del query, ls_path_score
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
query, ls_path_score = get_score(
    root_img_path, query_path, size, cosine_similarity)
plot_results(query_path, ls_path_score, reverse=True,
             method_name="cosine similarity")

query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
query, ls_path_score = get_score(
    root_img_path, query_path, size, cosine_similarity)
plot_results(query_path, ls_path_score, reverse=True,
             method_name="cosine similarity")

query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
query, ls_path_score = get_score(
    root_img_path, query_path, size, correlation_coefficient)
plot_results(query_path, ls_path_score, reverse=True,
             method_name="correlation coefficient")

query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
query, ls_path_score = get_score(
    root_img_path, query_path, size, correlation_coefficient)
plot_results(query_path, ls_path_score, reverse=True,
             method_name="correlation coefficient")
