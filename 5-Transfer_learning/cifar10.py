import numpy as np
import pickle
import os
import urllib.request
import tarfile
import zipfile

data_path = 'data/CIFAR-10/'
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# CIFAR-10 data seti 32*32 piksellik renkli(channels=3) resimlerden oluşmakta
# 10 farklı sınıfı var
img_size = 32
channels = 3
num_classes = 10

_train_files = 5
_images_per_file = 10000
_train_images = _train_files * _images_per_file

def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data

# CIFAR-10 verilerini tensörlere uyumlu olması için yeniden şekillendiriyoruz
def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, cls

# Dataset tarif edilen yerde bulunamazsa indir bulunursa yoksay
def _download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        def progress(block_num, block_size, total_size):
            progress_info = [url, float(block_num * block_size) / float(total_size) * 100.0]
            print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Done.")
    else:
        print("Data already exists.")

# sınıf bilgilerini one-hot olarak döndürüyor
def _one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]


def download():
    _download_and_extract(data_url, data_path)

# sınıf isimlerini yükle
def load_class_names():
    raw = _unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names

# eğitim verilerini yükle
def load_training_data():
    images = np.zeros(shape=[_train_images, img_size, img_size, channels], dtype=float)
    cls = np.zeros(shape=[_train_images], dtype=int)

    begin = 0
    for i in range(_train_files):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    return images, cls, _one_hot_encoded(class_numbers=cls, num_classes=num_classes)

# test verilerini yükle
def load_test_data():
    images, cls = _load_data(filename="test_batch")
    return images, cls, _one_hot_encoded(class_numbers=cls, num_classes=num_classes)