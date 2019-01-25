import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 로그출력 레벨설정
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

# 디렉토리 경로 정의
dirpath=[]
filename_path_list=[] # 파일 상대경로 리스트 (0~143) 프레임당 10개씩
filename_list=[] # 파일 이름 리스트 (0~143) 프레임당 10개씩
# 총 144 * 10 = 1440 개.
for a in range(0,144):
    dirpath.append("img_data\\img_data_rotate\\frame0000%d"%a)


# path : rotate 폴더 아래 디렉토리 반환
X_data = []
Y_data = []


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
Y_label = np.zeros((1440,2)) # Y_label One hot encoding

for i in range(0,len(Y_data),10):
    if Y_data[i] == '0':
        Y_label[i:i+10] = [1.0, 0.0]
    elif Y_data[i] == '1':
        Y_label[i:i+10] = [0.0, 1.0]

Y_data = Y_label

#finishd DataSet Setting

train_features = X_data[0:int(len(X_data))] # 특징 개수( 이미지 개수) * 0.8
train_labels = Y_data[0:int(len(Y_data))] # 라벨 개수( 이미지 라벨 개수) * 0.8

test_features = X_data[0:int(0.2*len(X_data))] # 테스트 특징 개수
test_labels = Y_data[0:int(0.2*len(Y_data))] #테스트 라벨 개수

# Training data declaration
'''
def train_data_iterator(): # 트레이닝 데이터 셔플후 반환.
    while True:
        idxs = np.arange(0, len(train_features))
        np.random.shuffle(idxs)
        shuf_features = train_features[idxs]
        shuf_labels=train_labels[idxs]
        batch_size = 10
        for batch_idx in range(0,len(train_features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch
'''
X = tf.placeholder(tf.float32,[None,6400])
# 1.자료형  2,데이터 크기에 맞출거면 None  3.입력 데이터크기
Y = tf.placeholder(tf.float32,[None,2])

# layer 1
w1 = tf.get_variable("w1",[6400,1024]) #in
b1 = tf.get_variable("b1",[1024]) # out
L1 = tf.nn.relu(tf.matmul(X, w1)+b1)

# layer 2
w2 = tf.get_variable("w2",[1024,256]) #in
b2 = tf.get_variable("b2",[256]) # out
L2 = tf.nn.relu(tf.matmul(L1, w2)+b2)

# layer 3
w3 = tf.get_variable("w3",[256,2])  #in shape
b3 = tf.get_variable("b3",[2]) #out shape
model = tf.add(tf.matmul(L2,w3),b3)

#Create the saver
saver = tf.train.Saver()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# cost 율 체크 , model에 label을 확인해봄으로서,(소프트맥스)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# cost가 최소화 될수있게 경사하강법을 이용해 가장 낮은 cost를 찾는 옵티마이저
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("==Training start===")
#100번 학습한다
a=0
for epoch in range(100):
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: train_features, Y: train_labels})
    # 트레이닝 과정의 cost_val 변화
    print("%d 번 학습의 Cost : %.6f"%(a,cost_val))
    a=a+1;
print("==Training finish===")
# 학습 된모델 저장
saver.save(sess, './model\\rotatemodel\\', global_step= 1000)
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

