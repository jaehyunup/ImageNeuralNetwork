import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 로그출력 레벨설정
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

# variable declaration
path = os.listdir("img_data\\img_data_r1")
# path : img_data\img_data_r1 하위 디렉토리(파일) 들을 모두반환
X_data = []
Y_data = []

# read image in X_data ( file read iteration and operating pixel 0~255 to 0~1)
for filename in path: # Path 리스트 크기만큼,순서대로
    file_path = os.path.join("img_data\img_data_r1", filename)
    img_read = cv2.imread(file_path, 2).ravel() / 255.0  # 0~255 => 0~1 치환
    X_data.append(img_read)

# read label in Y_data
label_File = open("img_data\\label.txt","r")
Y_data = label_File.read().splitlines()
# 라벨 데이터 입력

# array to np array and split training data and test data (shuffle)

Y_label = np.zeros((144,2)) # Y_label One hot encoding
for i in range(0, len(Y_data)):
    if Y_data[i] == '0':
        Y_label[i] = [1.0, 0.0]
    elif Y_data[i] == '1':
        Y_label[i] = [0.0, 1.0]
Y_data = Y_label

def shuffling(features,labels):
    c = list(zip(features, labels))
    shuffle(c)
    x_data, y_data = zip(*c)
    x_data = list(x_data)
    y_data = list(y_data)
    return np.array(x_data),np.array(y_data)

train_features = X_data[0:int(0.8*len(X_data))] # 특징 개수( 이미지 개수) * 0.8
train_labels = Y_data[0:int(0.8*len(Y_data))] # 라벨 개수( 이미지 라벨 개수) * 0.8

test_features = X_data[int(0.8*len(X_data)):] # 테스트 특징 개수
test_labels = Y_data[int(0.8*len(Y_data)):] #테스트 라벨 개수

# 학습시 사용되었던 모델과 동일하게 정의
X = tf.placeholder(tf.float32,[None,6400])
Y = tf.placeholder(tf.float32,[None,2])
# layer 1
w1 = tf.Variable(tf.random_normal([6400,1024],stddev = 0.01),name="w1") #in
b1 = tf.Variable(tf.zeros([1024]),name="b1") # out
L1 = tf.nn.relu(tf.matmul(X, w1)+b1)
# layer 2
w2 = tf.Variable(tf.random_normal([1024,256], stddev = 0.01),name="w2") #in
b2 = tf.Variable(tf.zeros([256]),name="b2") # out
L2 = tf.nn.relu(tf.matmul(L1, w2)+b2)
# layer 3
w3 = tf.Variable(tf.random_normal([256,2], stddev = 0.01),name="w3")  #in shape
b3 = tf.Variable(tf.zeros([2]),name="b3") #out shape
model = tf.add(tf.matmul(L2,w3),b3)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    # cost 율 체크 , model에 label을 확인해봄으로서,(소프트맥스)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    # cost가 최소화 될수있게 경사하강법을 이용해 가장 낮은 cost를 찾는 옵티마이저
    sess.run(init)
    print("==Training start===")
    #100번 학습한다
    a=0
    for epoch in range(2000):
        sp_train_features, sp_train_labels= shuffling(train_features,train_labels)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: sp_train_features, Y: sp_train_labels})
        # 트레이닝 과정의 cost_val 변화
        print("%d 번 학습의 Cost : %.6f"%(a,cost_val))
        a=a+1;
    print("==Training finish===")
    # 학습 된모델 저장
    #Create the save file
    saver.save(sess, './model\\'+"testModel", global_step=2000)
    print("==Model Saved OK.===")



print(test_features[0])


prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis = 1)
# 모델의 예측 비 계산
print('모델의 예측값', sess.run(prediction, feed_dict = {X: test_features}))
print('      실제 값', sess.run(target, feed_dict={Y: test_labels}))


# 정확도 계산
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %.2f' % sess.run(accuracy*100, feed_dict = {X: test_features, Y: test_labels}))

test_img_data = test_features[0:20]
fig = plt.figure()
for i in range(20):
    subplot = fig.add_subplot(4,5,i+1)
    subplot.imshow(test_img_data[i].reshape((80, 80)), cmap=plt.cm.gray_r)
plt.show()


