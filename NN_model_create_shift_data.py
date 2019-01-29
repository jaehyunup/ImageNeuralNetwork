import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 로그출력 레벨설정
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

def shuffling(features,labels):
    c = list(zip(features, labels))
    shuffle(c)
    x_data, y_data = zip(*c)
    x_data = list(x_data)
    y_data = list(y_data)
    return np.array(x_data),np.array(y_data)


# 디렉토리 경로 정의
dirpath=[]
filename_path_list=[] # 파일 상대경로 리스트 (0~143) 프레임당 10개씩
filename_list=[] # 파일 이름 리스트 (0~143) 프레임당 10개씩
# 총 144 * 10 = 1440 개.
for a in range(0,144):
    dirpath.append("img_data\\img_data_shift\\frame0000%d"%a)

print(dirpath)
# path : rotate 폴더 아래 디렉토리 반환
X_data = []
Y_data = []

#Create the saver

for framedir in dirpath:
    for root, dirs, files in os.walk(framedir):
        for fname in files:
            full_fname = os.path.join(root, fname)
            filename_list.append(fname)
            filename_path_list.append(full_fname) #디렉토리 내의 파일 name
            img_read = cv2.imread(full_fname, 2).ravel() / 255.0  # 0~255 => 0~1 치환
            X_data.append(img_read)

print(len(X_data)) # 1440개의 img . 10개씩 한프레임으로 취급.

# read label in Y_data
label_File = open("img_data\\label.txt","r")

Y_data = label_File.read().splitlines()
Y_label = []
print(len(Y_data))
#0~144 까지
for i in range(0,len(Y_data)):
    if Y_data[i] == '0':
        for j in range(10):
            #Y_label[i:i+10] = [1.0, 0.0]
            Y_label.append([1.0,0.0])
            #print("%d 번째 - 0 판단"%i)
            #print(len(Y_label[i:i+10]))
            #print(Y_label[i:i + 10])
    elif Y_data[i] == '1':
        for j in range(10):
            Y_label.append([0.0, 1.0])
            #Y_label[i:i+10] = [0.0, 1.0]
            #print("%d 번째 - 1 판단"%i)
            #print(len(Y_label[i:i+10]))
            #print(Y_label[i:i + 10])
#finishd DataSet Setting

Y_label = np.array(Y_label)

train_features = X_data[0:int(0.8*len(X_data))] # 특징 개수( 이미지 개수) * 0.8
train_labels = Y_label[0:int(0.8*len(Y_label))] # 라벨 개수( 이미지 라벨 개수) * 0.8

test_features = X_data[int(0.8*len(X_data)):] # 테스트 특징 개수
test_labels = Y_label[int(0.8*len(Y_label)):] #테스트 라벨 개수
print('t', train_labels)

#print(Y_label.shape)


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



saver = tf.train.Saver()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# cost 율 체크 , model에 label을 확인해봄으로서,(소프트맥스)
optimizer = tf.train.AdamOptimizer().minimize(cost)
# 값 원핫 인코딩
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    ckpt_path = saver.restore(sess, 'model/testModel-2000')
    # cost가 최소화 될수있게 경사하강법을 이용해 가장 낮은 cost를 찾는 옵티마이저
    #100번 학습
    a=0
    for epoch in range(1000):
        sp_train_features, sp_train_labels= shuffling(train_features,train_labels)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: train_features, Y: train_labels})
        #_, cost_val = sess.run([optimizer, cost], feed_dict={X: sp_train_features, Y: sp_train_labels})
        # 트레이닝 과정의 cost_val 변화
        print("%d 번 학습의 Cost : %.6f"%(a,cost_val))
        a=a+1;

    prediction = tf.argmax(model, axis=1)
    # 모델 원핫 인코딩
    target = tf.argmax(Y, axis=1)

    # 모델의 예측 비 계산
    print('모델의 예측값', sess.run(prediction, feed_dict={X: test_features}))
    print('      실제 값', sess.run(target, feed_dict={Y: test_labels}))

    # 정확도 계산
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy: %.2f' % sess.run(accuracy*100, feed_dict = {X: test_features, Y: test_labels}))
    print("==Training finish===")
    # 학습 된모델 저장
    saver.save(sess, './model\\shiftmodel\\shiftmodel', global_step= 1000)
    print("==Model Saved OK.===")


    # 데이터셋 체크
    print("==Check Original DataSet..===")
    print(train_labels[0:100:10])
    test_img_data = train_features[0:100:10]
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2,5,i+1)
        subplot.imshow(test_img_data[i].reshape((80,80)), cmap=plt.cm.gray_r)
    plt.show()

    # 데이터셋 체크
    print("==Check Test DataSet..===")
    print(test_labels[0:100:10])
    test_img_data = test_features[0:100:10]
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2,5,i+1)
        subplot.imshow(test_img_data[i].reshape((80,80)), cmap=plt.cm.gray_r)
    plt.show()



