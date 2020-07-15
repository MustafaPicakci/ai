import tensorflow as tf
 
 
import os
import re
import io
import requests
import numpy as np

from zipfile import ZipFile

data_dir='data/'
data_file='spam.txt'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#dataset yoksa indiriliyor
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')

    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))

#dataset varsa listeye atıyoruz
else:
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

#texti özel karakterlerden temizliyoruz
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return (text_string)

text_data_train = [clean_text(x) for x in text_data_train]

max_sequence_length = 25 #max text uzunluğu belirleyip uzun textleri parçalayacağız.
min_word_frequency= 5 #datasette 5 defadan daha az geçen kelimeleri yok sayacağız.
embedding_size = 50 #her kelime 50 boyutlu, eğitilebilir bir verktör içine gömülecek.
rnn_size=10

#Bu satırda texti vektörleştiriyoruz( = tokenleştirmek ? ) her kelimeye farklı bir değer veriliyor| örn: bugün hava yağmurlu = [0,1,2]
vocab_processor=tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                   min_frequency=min_word_frequency)
                                                                
text_processed=np.array(list(vocab_processor.fit_transform(text_data_train)))  

#burada elimizdeki datayı karıştırarak rastgeleliği artırıyoruz
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

#Bu kısımda dataseti bölerek eğitim ve test seti oluşturuyoruz
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)

x_data=tf.placeholder(tf.int32,[None,max_sequence_length])
y_true=tf.placeholder(tf.int32,[None])

# Makine öğreniminde embedding: Kelimeleri vektöre çeviriyoruz.Vektörler sayılardan oluştuğu için
# onlar üzerinde hesaplamalar yapabiliyoruz. Word embedding ile iyi bir şekilde eğitilmiş modeller
# dili anlayabiliyor ve benzer kelimeler arasında bağlantılar kurabiliyor.

# embedding matris oluşturup içini -1 ile 1 arasında random sayılarla dolduruyoruz.
embedding_mat=tf.Variable(tf.randomm_uniform([vocab_size,embedding_size],-1.0,1.0))
embedding_output=tf.nn.embedding_lookup(embedding_mat,x_data) #Id ye göre parametre döndürüyor.
# param=[10,20,30,40]  id=[2,0,3,1]  embedded_lookup(param,id)=[30,10,40,20]

cell=tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
output,state=tf.nn.dynamic_rnn(cell,embedding_output,dtype=tf.float32) # while döngüsü kullanarak hücreden gerektiği kadar geçiş yapacak. Çok boyutlu tensor döndürecek

output=tf.transpose(output,[1,0,2])# output'u istediğimiz şekle getiriyoruz
last=tf.gather(output,int(output.get_shape()[0])-1) # gather ile ihtiyacımız olan yeri alıyoruz

w=tf.Variable(tf.truncated_normal([rnn_size,2],stddev=0.1))# tahminleri alabilmek için fully connected output layerı oluşturuyoruz| 2nin sebebi tahmin spam veya hun(normal) olduğu için
b=tf.Variable(tf.constant(0.1,shape=[2]))

logits=tf.matmul(last,w) + b
y=tf.nn.softmax(logits)

xent=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,label=y_true)
loss=tf.reduce_mean(xent)

correct=tf.equal(tf.argmax(y,1),tf.cast(y_true,tf.int64)) #tf.equal ile doğruları alıyoruz| y_true'yu tf.cast ile integer a dönüştürüyoruz.
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

optimizer=tf.train.AdamOptimizer(1e-3).minimize(loss) #0.001

sess=tf.Session()
sess.run(tf.global_variables_initializer())

batch_size=250

def train_step(epochs): # datasetin tamamının bir defa eğitimden geçmesine epochs denir.
    for epoch in range(epochs):
        shuffled_ix=np.random.permutation(np.arange(len(x_train)))# aynı eğitim setinin üzerinden sürekli geçilecekse eğitim setinin karıştırılması önerilir. eğitim setini karıştırıoyruz
        x=x_train[shuffled_ix]
        y=y_train[shuffled_ix]

        n_batches=len(x) // batch_size + 1 # // sonucu int istediğimiz anlamına geliyor| tüm dataseti aynı anda beslemek istiyoruz

        for i in range(n_batches): # tüm dataseti baştan sona eğitiliyor. Dahafazla eğitilmek istenirse dataset karıştırılıp yeniden bu döngüye girecek
            min_ix=i*batch_size
            max_ix=np.min([len(x),((i+1) * batch_size)]) #datasertin uzunluğu aşılmasın diye güvenlik önlemi
            x_train_batch=x[min_ix:max_ix] #batch halinde dataseti aldık
            y_train_batch=y[min_ix:max_ix]

            feed_dict_train={x_data:x_train_batch,y_true:y_train_batch} # feed_dict ile placeholderlara atama yapıyoruz
            sess.run(optimizer, feed_dict=feed_dict_train)

        [train_loss,train_acc]=sess.rus([loss,accuracy],feed_dict=feed_dict_train)
        print('Epoch: ',epoch+1, 'Training accuracy: ',train_acc, 'Training loss: ',train_loss)

def test_step():
    feed_dict_test={x_data : x_test,y_true : y_test}
    acc= sess.run(accuracy,feed_dict=feed_dict_test)
    print('Testing accuracy: ',acc)


train_step(20)
test_step()