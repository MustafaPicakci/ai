import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta

import cifar10

cifar10.download()

print(cifar10.load_class_names())

train_img, train_cls, train_labels=cifar10.load_training_data()
test_img, test_cls, test_labels=cifar10.load_test_data()

print(len(train_img),len(test_img))
x=tf.placeholder(tf.float32,[None,32,32,3]) # 32*32 lik ve derinliği 3 olan(RGB- Renkli) görseller kullanıyoruz
y_true=tf.placeholder(tf.float32,[None,10]) # 10 adet sınıf sayımız var çıkış 10 sınıftan biri olacak
pkeep=tf.placeholder(tf.float32)

def pre_process_image(image):
    image=tf.image.random_flip_left_right(image)    #resimleri ters çeviriyoruz
    image=tf.image.random_hue(image,max_delta=0.05) #resimlerin renkleri ile oynanıyor
    image=tf.image.random_contrast(image,lower=0.3,upper=1.0) #resimlerin kontrast değerleri ile oynuyoruz
    image=tf.image.random_brightness(image,max_delta=0.2)      #resimlerin parlaklık değerleri ile oynuyoruz
    image=tf.image.random_saturation(image,lower=0.0,upper=2.0)  #resimlerin doygunluk değerleriyle oynuyoruz

    image=tf.minimum(image,1.0)    # işlemlerin geçerli değerler arasında uygulanması için
    image=tf.maximum(image,0.0)    # işlemlerin geçerli değerler arasında uygulanması için
    return image

def pre_process(images):
    images=tf.map_fn(lambda image : pre_process_image(image),images) #tf.map_fn() çağırılan fonksiyonları tüm elemanlara uygular
    return images

with tf.device('/cpu:0'): # pre process gpuda çok yavaş çalıştığından bu işlemi cpuda çalıştırıyoruz
    distorted_images=pre_process(images=x) # x resimlerini data augmentationdan geçiriyoruz

def conv_layer(input,size_in,size_out,use_pooling=True): # layerları otomatik oluşturmak için fonksiyon yazıyoruz
    w=tf.Variable(tf.truncated_normal([3,3,size_in,size_out],stddev=0.1)) #filtre boyutu 3x3 | size_in:önceki layerın derinliği | size_out: filtre sayısı
    b=tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv=tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME')+b #strides ile filtrenin hareketini ayarladık | padding'same' boyutu korumak için
    y=tf.nn.relu(conv)

    # maxpooling yaparak eğitilecek parametre sayısını düşürüyoruz(Eğer use_pooling true ise)
    if use_pooling:
        y=tf.nn.max_pool(y,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #ksize ile 2x2 boyutunda ve derinliği 1 olan pencere oluşturduk    
                                                                              #çakışma olmasın diye strides ile 2-2 kaydırıyoruz
        return y

#fully connected layer oluşturmak için fonksiyon oluşturduk
def fc_layer(input,size_in,size_out,relu=True, dropout=True):
    w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1))
    b=tf.Variable(tf.constant(0.1, shape=[size_out]))
    logits=tf.matmul(input,w)+b

    if relu:
        y= tf.nn.relu(logits)
        if dropout:
            y=tf.nn.dropout(y,pkeep)
            return y
    else:
        return logits

conv1=conv_layer(distorted_images,3,32,use_pooling=True) #3 önceki layerin derinliği| 32 filtre sayısı output:16x16x32 olacak
conv2=conv_layer(conv1,32,64,use_pooling=True) #output:8x8x64 olacak
conv3=conv_layer(conv2,64,64,use_pooling=True) #output:4x4x64 olacak
flattened=tf.reshape(conv3,[-1,4*4*64]) #convlayeri fullyconnected layera bağlayabilmek için düzleştirmemiz gerekiyor. bunu reshape ile yapıyoruz
fc1=fc_layer(flattened,4*4*64,512,relu=True,dropout=True) #512 layerdeki nöron sayısına karşılık geliyor
fc2=fc_layer(fc1,512,256,relu=True,dropout=True) #256 layerdeki nöron sayısına karşılık geliyor
logits=fc_layer(fc2,256,10,relu=False,dropout=False) #10 layerdeki nöron sayısına karşılık geliyor | output layeri
y=tf.nn.softmax(logits)

y_pred_cls=tf.argmax(y,1) #tahmin edilen sınıfı bir değişkene atıyoruz
xent=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
loss=tf.reduce_mean(xent)

correct_prediction=tf.equal(y_pred_cls,tf.argmax(y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

optimizer=tf.train.AdamOptimizer(5e-4).minimize(loss) # learning rate=0.0005

sess=tf.Session()
sess.run(tf.global_variables_initializer())

batch_size=128


def random_batch():
    index=np.random.choice(len(train_img),size=batch_size,replace=False) #Rastgele index oluşturup bu indexteki resimleri ve labelleri alacağız.
    x_batch=train_img[index, :, :, :] #resimleri bu değişkene atadık
    y_batch=train_labels[index, :] #labelleri bu değişkene atadık.
    return x_batch,y_batch

loss_graph=[]

def training_step(iterations):
    start_time=time.time()
    for i in range(iterations):
        x_batch,y_batch=random_batch()
        feed_dict_train={x:x_batch, y_true:y_batch,pkeep:0.5} #pkeep ile aktif olacak nöron miktarını belirliyoruz
        [_,train_loss]=sess.run([optimizer,loss],feed_dict=feed_dict_train) #loss değerini değişkene atadık  
        loss_graph.append(train_loss) # loss değerini listeye ekledik. bu değerler ile loss grafiği oluşturacağız.

        if i % 100 == 0:
            acc=sess.run(accuracy,feed_dict=feed_dict_train)
            print('Iteration: ',i,' Train accuracy: ',acc,' Training loss:',train_loss)
        
    end_time=time.time()
    time_dif=end_time-start_time
    print('Time Usage: ',timedelta(seconds=int(round(time_dif))))


batch_size_test=256
def test_accuracy():
    num_images=len(test_img) #test setindeki resim sayısını değişkene atadık
    cls_pred=np.zeros(shape=num_images,dtype=np.int) #boyutu num_images kadar olan ve sıfırlardan oluşan bir liste oluşturduk.
                                                     #bu listeye tahmin ettiğimiz sınıfları atayacağız
    i=0
    while i <num_images: # bütün resimler bitene kadar bu döngü çalışacak
        j=min(i + batch_size_test,num_images) # j her seferinde 256 artacak ve num_images'i geçmeyecek| min fonksiyonu ile bu ikisinden küçük olan alınıyor
        feed_dict={x: test_img[i:j, :], y_true: test_labels[i:j, :],pkeep:1} #i den j ye kadar image ve labelleri alıyoruz
        cls_pred[i:j]=sess.run(y_pred_cls,feed_dict=feed_dict) #tahmin edilen sınıfları listeye atıyoruz
        i=j
# i den j ye kadar olan resimleri alıp bunları modelimize tahmin ettiriyoruz. Tahmin ettiği sınıfları bir listeye atıyoruz
# en son i ye j yi atayarak i yi 256 artırıyoruz ve tüm resimler bitene kadar döngü çalışıyor

        correct=(test_cls== cls_pred)   # tahminler ile doğru cevapları karşılaştırıyoruz
    print('Testing accuracy: ',correct.mean()) # ortalamasını aldık

training_step(10000)
test_accuracy()

plt.plot(loss_graph,'k-')
plt.title('Loss Grafiği')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show() 
