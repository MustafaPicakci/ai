import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import os
import sys
import tarfile
import zipfile
import urllib.request
import pickle


data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
data_path = "data/inception/"
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"
path_graph_def = "classify_image_graph_def.pb"

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

def download():
    _download_and_extract(data_url, data_path)


def cache(cache_path, fn, *args, **kwargs):
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        obj = fn(*args, **kwargs)
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


class NameLookup:
    def __init__(self):
        self._uid_to_cls = {}
        self._uid_to_name = {}
        self._cls_to_uid = {}

        path = os.path.join(data_path, path_uid_to_name)
        with open(file=path, mode='r') as file:
            lines = file.readlines()

            for line in lines:
                line = line.replace("\n", "")
                elements = line.split("\t")
                uid = elements[0]
                name = elements[1]
                self._uid_to_name[uid] = name

        path = os.path.join(data_path, path_uid_to_cls)
        with open(file=path, mode='r') as file:
            lines = file.readlines()

            for line in lines:
                if line.startswith("  target_class: "):
                    elements = line.split(": ")
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    elements = line.split(": ")
                    uid = elements[1]
                    uid = uid[1:-2]
                    self._uid_to_cls[uid] = cls
                    self._cls_to_uid[cls] = uid

    def uid_to_cls(self, uid):
        return self._uid_to_cls[uid]

    def uid_to_name(self, uid, only_first_name=False):
        name = self._uid_to_name[uid]
        if only_first_name:
            name = name.split(",")[0]

        return name

    def cls_to_name(self, cls, only_first_name=False):
        uid = self._cls_to_uid[cls]
        name = self.uid_to_name(uid=uid, only_first_name=only_first_name)

        return name


class Inception:
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"
    tensor_name_input_image = "DecodeJpeg:0"
    tensor_name_resized_image = "ResizeBilinear:0"
    tensor_name_softmax = "softmax:0"
    tensor_name_softmax_logits = "softmax/logits:0"
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):
        self.name_lookup = NameLookup()
        self.graph = tf.Graph()
        with self.graph.as_default():
            path = os.path.join(data_path, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

        self.y_pred = self.graph.get_tensor_by_name(self.tensor_name_softmax)
        self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)
        self.resized_image = self.graph.get_tensor_by_name(self.tensor_name_resized_image)
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)
        self.transfer_len = self.transfer_layer.get_shape()[3]

        self.session = tf.Session(graph=self.graph)

    def close(self):
        self.session.close()

    def _write_summary(self, logdir='summary/'):
        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def _create_feed_dict(self, image_path=None, image=None):
        if image is not None:
            feed_dict = {self.tensor_name_input_image: image}

        elif image_path is not None:
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            feed_dict = {self.tensor_name_input_jpeg: image_data}

        else:
            raise ValueError("Either image or image_path must be set.")

        return feed_dict

    def classify(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        pred = self.session.run(self.y_pred, feed_dict=feed_dict)
        pred = np.squeeze(pred)

        return pred

    def get_resized_image(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)
        resized_image = resized_image.squeeze(axis=0)
        resized_image = resized_image.astype(float) / 255.0

        return resized_image

    def print_scores(self, pred, k=10, only_first_name=True):
        idx = pred.argsort()
        top_k = idx[-k:]
        for cls in reversed(top_k):
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)
            score = pred[cls]

            print("{0:>6.2%} : {1}".format(score, name))

    def transfer_values(self, image_path=None, image=None):
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)
        transfer_values = np.squeeze(transfer_values)

        return transfer_values



def process_images(fn, images=None, image_paths=None):
    using_images = images is not None
    if using_images:
        num_images = len(images)
    else:
        num_images = len(image_paths)
    result = [None] * num_images
    for i in range(num_images):
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)
        sys.stdout.write(msg)
        sys.stdout.flush()
        if using_images:
            result[i] = fn(image=images[i])
        else:
            result[i] = fn(image_path=image_paths[i])

    print()

    result = np.array(result)

    return result



def transfer_values_cache(cache_path, model, images=None, image_paths=None):
    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)
    transfer_values = cache(cache_path=cache_path, fn=fn)

    return transfer_values



if __name__ == '__main__':
    download()
    model = Inception()
    image_path = os.path.join(data_path, 'cropped_panda.jpg')
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10)
    model.close()
