import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import cifar10
from cifar10 import num_classes
import inception
from inception import transfer_values_cache

cifar10.download()
inception.download()

model=inception.Inception()
print('model alındı')
train_img,train_cls,train_labels=cifar10.load_training_data()
test_img,test_cls,test_labels=cifar10.load_test_data()

file_path_cache_train=os.path.join(cifar10.data_path,'inception_cifar10_train.pkl')
file_path_cache_test=os.path.join(cifar10.data_path,'inception_cifar10_test.pkl')

images_scaled=train_img * 255.0 #inception'da renk değerleri 0-255 aralığında,
                                #cifar10'da ise 0-1 arasında bu yüzden 255 ile çarparak aynı aralığa getiriyoruz

transfer_values_train=transfer_values_cache(cache_path=file_path_cache_train, #transfer değerleri daha önceden hesaplanmış ise yükleniyor,
                                            images=images_scaled,             #hesaplanmamışsa hesaplanıyor ve dosyaya yazılıyor. 
                                            model=model) 
                                                                             
images_scaled=test_img * 255.0 #inception'da renk değerleri 0-255 aralığında,
                                #cifar10'da ise 0-1 arasında bu yüzden 255 ile çarparak aynı aralığa getiriyoruz

transfer_values_test=transfer_values_cache(cache_path=file_path_cache_test, #transfer değerleri daha önceden hesaplanmış ise yükleniyor,
                                            images=images_scaled,             #hesaplanmamışsa hesaplanıyor ve dosyaya yazılıyor. 
                                            model=model) 

# print(transfer_values_train.shape())     #50000 x 2048  '50000' resim sayısı, '2048' gelen transfer değerleri

x=tf.placeholder(tf.float32,[None,2048])
y_true=tf.placeholder(tf.float32,[None,num_classes])
                                            #1025 nöron sayısı
weight1=tf.Variable(tf.truncated_normal([2048,1024],stddev=0.1))  # fully connected layer tanımı
bias1=tf.Variable(tf.constant(0.1,shape=[1024]))                
weight2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))  # output layer tanımı
bias2=tf.Variable(tf.constant(0.1,shape=[10]))

y1=tf.nn.relu(tf.matmul(x,weight1)+ bias1)  #Nöronları bağlıyoruz
logits=tf.matmul(y1,weight2)+bias2 #output layeri
y2=tf.nn.softmax(logits)

y_pred_cls=tf.argmax(y2,1) #tahmin edlien sınıf için değişken oluşturduk

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
def random_batch():                                     #Rastgele index oluşturup bu indexteki resimleri ve labelleri alacağız.
    num_images = len(transfer_values_train)
    idx = np.random.choice(num_images, size=batch_size, replace=False)
    x_batch = transfer_values_train[idx]    #resimleri bu değişkene atadık
    y_batch = train_labels[idx]             #labelleri bu değişkene atadık.

    return x_batch, y_batch

loss_graph = []
def training_step (iterations):
    start_time = time.time()
    for i in range (iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', acc, 'Training loss:', train_loss)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: ", timedelta(seconds=int(round(time_dif))))


batch_size_test = 256
def test_accuracy():
    num_images = len(transfer_values_test)              #test setindeki resim sayısını değişkene atadık
    cls_pred = np.zeros(shape=num_images, dtype=np.int) #boyutu num_images kadar olan ve sıfırlardan oluşan bir liste oluşturduk.
                                                        #bu listeye tahmin ettiğimiz sınıfları atayacağız

    i = 0
    while i < num_images:
        j = min(i + batch_size_test, num_images)    # j her seferinde 256 artacak ve num_images'i geçmeyecek| min fonksiyonu ile bu ikisinden küçük olan alınıyor
        feed_dict = {x: transfer_values_test[i:j],
                     y_true: test_labels[i:j]}      #i den j ye kadar transfer value ve labelleri alıyoruz
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (test_cls == cls_pred)
    print('Testing accuracy:', correct.mean())

training_step(10000)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title('Loss grafiği')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()