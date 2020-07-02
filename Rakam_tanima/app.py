import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/MNIST/",one_hot=True)# 10 tane çıkıştan sadece bir tanesi 1 ile gösterilerek tahmin sonucu gösterilecek(örn 8={0,0,0,0,0,0,0,0,1,0})

#input
x=tf.placeholder(tf.float32,[None,784]) # None gelen resimlerin sayısı. Şu an her geleni alacak. 784 boyut
                                        # x input. x'e mnist resimlerinin renk değerlerini atadık
#label
y_true=tf.placeholder(tf.float32,[None,10]) # 10 dememizin sebebi sonuçta 10 tane çıkış olaması (0-9 arasındaki rakamlar)

w=tf.Variable(tf.zeros([784,10]))   # y=x.w+b
b=tf.Variable(tf.zeros([10]))       # w ve b model eğitilirken otamatik optimize edilecek değerler

logist=tf.matmul(x,w) + b           # matmul metodu ile matrislerde çarpma işlemi gerçekleştiriyoruz.
y=tf.nn.softmax(logist)             # nörondaki değerleri 0-1 arasına çekiyor. Tüm bu değerlerin toplamı 1'e eşit oluyor. (olasılık gibi)

xent=tf.nn.softmax_cross_entropy_with_logits(logits=logist,labels=y_true)    # loss değeri hesaplıyoruz
loss=tf.reduce_mean(xent)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1) )   # argmax(y,1) ile tahmini alıyoruz. y tahmin, 1 boyut | argmax(y_true,1) ile gerçek değeri alıyoruz
                                                                   # argmax fonksiyonu vektör içerisindeki en yüksek değerin kaçıncı sırada olduğu bilgisini veriyor.
                                                                   # tf.equal metodu ile tahmin ve gerçek değeri karşılaştırıyoruz. 
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    # correct_prediction'ın ortalamasını alarak modelin % kaç doğru tahmin ettiğini bulacağız 

optimize=tf.train.GradientDescentOptimizer(0.5).minimize(loss)                    # yazılan sayı learning rate. Burada loss'u azaltarak optimize ediyoruz                                                               

sess=tf.Session()
sess.run(tf.global_variables_initializer())
batch_size=128  # resimlerin tek seferde kaç tanesi işlenecek


def training_step(iterations):
    for i in range(iterations):
        x_batch,y_batch=mnist.train.next_batch(batch_size)         #x_batch'e resimleri, y_batch'e etiketleri atadık
        feed_dict_train = {x : x_batch, y_true : y_batch}
        sess.run(optimize,feed_dict=feed_dict_train)



def test_accuracy():
    feed_dict_test={x: mnist.test.images, y_true: mnist.test.labels}
    acc=sess.run(accuracy,feed_dict=feed_dict_test)
    print("testing accuracy : ", acc)


training_step(2000)
test_accuracy()