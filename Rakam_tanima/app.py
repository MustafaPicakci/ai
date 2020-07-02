import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/MNIST/",one_hot=True)# 10 tane çıkıştan sadece bir tanesi 1 ile gösterilerek tahmin sonucu gösterilecek(örn 8={0,0,0,0,0,0,0,0,1,0})

#input
x=tf.placeholder(tf.float32,[None,784]) # None gelen resimlerin sayısı. Şu an her geleni alacak. 784 boyut
                                        # x input. x'e mnist resimlerinin renk değerlerini atadık
#label
y_true=tf.placeholder(tf.float32,[None,10]) # 10 dememizin sebebi sonuçta 10 tane çıkış olaması (0-9 arasındaki rakamlar)

#dropout sabiti
pkeep=tf.placeholder(tf.float32)    # dropout ile eğitim esnasında random olarak bazı nöronları inaktif edip modelin daha iyi genelleme yapmasını
                                    # Sağlayacağız.

#layers
layer_1=128     # verilen sayılar layerlerdeki nöron sayıları.(istediğimiz gibi ayarlayabiliyoruz.)
layer_2=64
layer_3=32
layer_out=10    # Çıkış layerindeki nöron sayısı. (0-10 arasındaki rakamların her biri için bir nöron eklendi)

#w=tf.Variable(tf.zeros([784,10]))   # y=x.w+b
#b=tf.Variable(tf.zeros([10]))       # w ve b model eğitilirken otamatik optimize edilecek değerler

weight_1 = tf.Variable(tf.truncated_normal([784,layer_1],stddev=0.1)) #rastgele küçük bir sayı atanıyor | stddev: standart sapma
bias_1=tf.Variable(tf.constant(0.1,shape=[layer_1]))
weight_2 = tf.Variable(tf.truncated_normal([layer_1,layer_2],stddev=0.1)) #rastgele küçük bir sayı atanıyor | stddev: standart sapma
bias_2=tf.Variable(tf.constant(0.1,shape=[layer_2]))
weight_3 = tf.Variable(tf.truncated_normal([layer_2,layer_3],stddev=0.1)) #rastgele küçük bir sayı atanıyor | stddev: standart sapma
bias_3=tf.Variable(tf.constant(0.1,shape=[layer_3]))
weight_4 = tf.Variable(tf.truncated_normal([layer_3,layer_out],stddev=0.1)) #rastgele küçük bir sayı atanıyor | stddev: standart sapma
bias_4=tf.Variable(tf.constant(0.1,shape=[layer_out]))

y1=tf.nn.relu(tf.matmul(x,weight_1)+bias_1)
y1d=tf.nn.dropout(y1,pkeep)
y2=tf.nn.relu(tf.matmul(y1d,weight_2)+bias_2)
y2d=tf.nn.dropout(y2,pkeep)
y3=tf.nn.relu(tf.matmul(y2d,weight_3)+bias_3)
y3d=tf.nn.dropout(y3,pkeep)
logits=tf.matmul(y3d,weight_4)+ bias_4
y4=tf.nn.softmax(logits)

xent=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
loss=tf.reduce_mean(xent)

correct_prediction=tf.equal(tf.argmax(y4,1),tf.argmax(y_true,1) )  # argmax(y,1) ile tahmini alıyoruz. y tahmin, 1 boyut | argmax(y_true,1) ile gerçek değeri alıyoruz
                                                                   # argmax fonksiyonu vektör içerisindeki en yüksek değerin kaçıncı sırada olduğu bilgisini veriyor.
                                                                   # tf.equal metodu ile tahmin ve gerçek değeri karşılaştırıyoruz. 
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    # correct_prediction'ın ortalamasını alarak modelin % kaç doğru tahmin ettiğini bulacağız 
optimize=tf.train.AdamOptimizer(0.001).minimize(loss)     # yazılan sayı learning rate. Burada loss'u azaltarak optimize ediyoruz                                                               

sess=tf.Session()
sess.run(tf.global_variables_initializer())
batch_size=128  # resimlerin tek seferde kaç tanesi işlenecek
loss_graph= []

#logist=tf.matmul(x,w) + b           # matmul metodu ile matrislerde çarpma işlemi gerçekleştiriyoruz.
#y=tf.nn.softmax(logist)             # nörondaki değerleri 0-1 arasına çekiyor. Tüm bu değerlerin toplamı 1'e eşit oluyor. (olasılık gibi)

#xent=tf.nn.softmax_cross_entropy_with_logits(logits=logist,labels=y_true)    # loss değeri hesaplıyoruz
#loss=tf.reduce_mean(xent)

#correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1) )   # argmax(y,1) ile tahmini alıyoruz. y tahmin, 1 boyut | argmax(y_true,1) ile gerçek değeri alıyoruz
                                                                   # argmax fonksiyonu vektör içerisindeki en yüksek değerin kaçıncı sırada olduğu bilgisini veriyor.
                                                                   # tf.equal metodu ile tahmin ve gerçek değeri karşılaştırıyoruz. 
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    # correct_prediction'ın ortalamasını alarak modelin % kaç doğru tahmin ettiğini bulacağız 

#optimize=tf.train.GradientDescentOptimizer(0.5).minimize(loss)                    # yazılan sayı learning rate. Burada loss'u azaltarak optimize ediyoruz                                                               

#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
#batch_size=128  # resimlerin tek seferde kaç tanesi işlenecek


def training_step(iterations):
    for i in range(iterations):
        x_batch,y_batch=mnist.train.next_batch(batch_size)      #x_batch'e resimleri, y_batch'e etiketleri atadık
        feed_dict_train = {x : x_batch, y_true : y_batch, pkeep:0.75} #pkeep:0.75 ile nöronların %75i aktif olacak anlamına geliyor.
        #sess.run(optimize,feed_dict=feed_dict_train)                   dropout için kullanıyoruz.
        [_,train_loss]=sess.run([optimize,loss],feed_dict=feed_dict_train)   #optimize ederken loss değerinin düşüp düşmediğini gözlemlemek için atama yaptık
        
        loss_graph.append(train_loss)

        if i % 100 == 0:    #her yüz adımda yazdırıyoruz
            train_acc=sess.run(accuracy,feed_dict=feed_dict_train)
            print('Iterations:',i, 'Training accuracy:',train_acc, 'Training loss:', train_loss)


feed_dict_test={x: mnist.test.images, y_true: mnist.test.labels, pkeep: 1}
def test_accuracy():  
    acc=sess.run(accuracy,feed_dict=feed_dict_test)
    print("testing accuracy : ", acc)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])



#training_step(2000)
training_step(10000)
test_accuracy()

plt.plot(loss_graph,'k-')
plt.title('Loss Grafiği')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

plot_example_errors() #modeliminizin tahmin etkmekte zorlandığı 9 resmi ekrana yazdırarak inceliyoruz