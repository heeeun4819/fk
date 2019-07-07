####
## 2019.07.06 Yang Hee Eun
####

from getEmbeddings import dat_embedding
import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, Lambda, Input

sess = tf.Session()
K.set_session(sess)

# ELMo 다운로드
elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

#데이터 불러오기
# Read the data
if not os.path.isfile('./xtr.npy') or \
    not os.path.isfile('./xte.npy') or \
    not os.path.isfile('./ytr.npy') or \
    not os.path.isfile('./yte.npy'):
    xtr,xte,ytr,yte = dat_embedding("datasets/train.csv")
    np.save('./xtr', xtr)
    np.save('./xte', xte)
    np.save('./ytr', ytr)
    np.save('./yte', yte)
print('====================')
xtr = np.load('./xtr.npy')  #16608
xte = np.load('./xte.npy')  #4153
ytr = np.load('./ytr.npy')  #16608
yte = np.load('./yte.npy')  #4153

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]


input_text = Input(shape=(300,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
print("************************")
print(input_text.shape) # (?,300)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#print(model.summary())
print(xtr.shape) #x-train shape
print(ytr.shape) #y-train shape
print(xte.shape) #x-test shape
print(yte.shape) #x-test shape
model.fit(xtr, ytr, validation_data=(xte, yte), epochs=5, batch_size=64)
scores = model.evaluate(xte, yte, verbose=0)

print("\n 테스트 정확도: %.4f" % (scores[1]*100))

