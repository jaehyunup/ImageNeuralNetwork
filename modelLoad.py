import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from cv2 import cv2
except ImportError:
    pass
import tensorflow as tf
import numpy as np

def img_trim(src_image):
    src_real=src_image.copy()
    temp_list=[]
    x ,y=0,0
    for i in range(0,16):
        for j in range(0,9):
            src_temp=np.array(src_real[j*80:(j+1)*80,i*80:(i+1)*80]).ravel()
            temp_list.append(src_temp)

    return np.array(temp_list)

# 학습시 사용되었던 모델과 동일하게 정의
X = tf.placeholder(tf.float32,[None,6400])
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
saver = tf.train.Saver()

#get model data
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint("model\\shiftmodel"))

prediction = tf.argmax(model, axis = 1)
# 모델 원핫 인코딩
target = tf.argmax(Y, axis = 1)
# 값 원핫 인코딩

cap = cv2.VideoCapture('img_data\\car3.mp4')
ret, frame = cap.read()  # binary Video 객체
# start
while (ret ==1) :
    grayframe = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    #cv2.imshow('trim', img_trim(grayframe ,0, 80, 0, 80))
    grayframe_trim = img_trim(grayframe)
    #window=cv2.imshow("image",grayframe_trim[5]) # 144개의 trim image 가 있음.3 차원 ndarray.
    # grayframe_trim[0~143] = 지금 프레임에서 trim 되어 갈라진 각 80x80 한 픽셀을 의미함
    test_feature=grayframe_trim / 255.0  # 0~255 => 0~1 치환
    # 모델의 예측 비 계산
    prelist=sess.run(prediction, feed_dict = {X: test_feature})
    #print("모델 예측값", prelist[3:4,])
    #print(grayframe_trim.shape)
    prelist=np.array(prelist)
    width,height=0,0
    count=0

    # 배경영역 색칠(144개의 픽셀다 돌아야함)
    for xi in range (0,16):
        # 픽셀이 배경일때 1 , 전경일때 0 .
        for xj in range(0,9):
            if prelist[count]==1:
                #print(prelist[count])
                grayframe[xj*80:(xj+1)*80,xi*80:(xi+1)*80]= 255
            count = count + 1

    cv2.imshow('pre',grayframe)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('p'):
        while (1000):
            wk = cv2.waitKey(0)
            if wk == ord('p'):
                break
    ret, frame = cap.read()  # binary Video 객체
cap.release()
cv2.destroyAllWindows()