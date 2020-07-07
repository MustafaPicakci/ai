import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf    
tf.disable_v2_behavior()
import numpy as np

import inception

inception.download()

def conv_layer_names (): # Inception modelindeki conv layerlerin isimlerini alacak fonksiyon
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D'] #sadece conv2d türündeki layerleri alıyoruz
    model.close()
    return names

conv_names = conv_layer_names()

def optimize_image(conv_id=0,feature=0,iterations=30,show_progress=True):
    model=inception.Inception()
    resized_image=model.resized_image

    conv_name = conv_names[conv_id]
    tensor = model.graph.get_tensor_by_name(conv_name + ":0")

    with model.graph.as_default():
        loss = tf.reduce_mean(tensor[:, :, :, feature]) #loss değeribelirtilen featurenin lost değerlerinin ortalaması

    gradient = tf.gradients(loss, resized_image)
    sess = tf.Session(graph=model.graph)

    image_shape=resized_image.get_shape() #resized image boyutunu aldık
    image=np.random.uniform(size=image_shape)+128.0 #128 resim etrafında rastgele renkler verecek

    for i in range(iterations):
        feed_dict={model.tensor_name_resized_image:image }
        [grad,loss_value]=sess.run([gradient,loss],feed_dict=feed_dict)
        grad=np.array(grad).squeeze() # grad artık bize input resmini ne kadar değiştirirsek feature'ı maksimize ederiz onu gösteriyor

        step_size=1.0/(grad.std()+1e-8) #1e-8 yazmamızın sebebi sıfıra bölünmesinin engellemek
        image += step_size * grad
        image=np.clip(image,0.0,255.0) #np.clip ile resmin renk değerlerinin 0-255 arasına sınırlıyoruz

        if show_progress:
            print('Iterations : ',i)
            msg="Gradient min : {0:>9.6f}, max:{1:>9.6f}, stepsize:{2:9.2f}"
            print("loss: ",loss_value)
        
    model.close()
    return image.squeeze()
 
def normalize_image(x): #resmin değerlerinin 0-1 arasına sıkıştırmak için
    x_min=x.min()
    x_max=x.max()

    x_norm= (x - x_min) / (x_max - x_min)

    return x_norm

def plot_image(image): #resimleri yazdırmak için
    img_norm=normalize_image(image)
    plt.imshow(img_norm)
    plt.show()

image=optimize_image(conv_id=1,feature=5,iterations=30)
plot_image(image) 