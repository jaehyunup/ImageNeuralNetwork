import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from cv2 import cv2
except ImportError:
    pass
import tensorflow as tf
import matplotlib

#image trim function
def img_trim(src_image):
    temp_list=[]
    src_real=src_image.copy()
    x,y=80,80
    #16대 9의 영상비를 가짐
    for i in range(0,16):
        x=x+80
        for j in range(0,9):
            src_real[0:x,0:y]
            y=y+80
            temp_list.append(src_real)
            if j==8 :
                y=80
    return temp_list

cap = cv2.VideoCapture('img_data\\car3.mp4')
ret, frame = cap.read()  # binary Video 객체
grayframe = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
#cv2.imshow('trim', img_trim(grayframe ,0, 80, 0, 80))
grayframe_trim = img_trim(grayframe)

print(len(grayframe_trim)) # 144개의 trim image



k = cv2.waitKey(0) & 0xff
cap.release()
cv2.destroyAllWindows()


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
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint("model"))
    #print("Model restored.")
    #print(w2.eval())


prediction = tf.argmax(model, axis = 1)
# 모델 원핫 인코딩
target = tf.argmax(Y, axis = 1)
# 값 원핫 인코딩
'''
# 모델의 예측 비 계산
print('모델의 예측값', sess.run(prediction, feed_dict = {X: test_features}))

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
'''
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# cost 율 체크 , model에 label을 확인해봄으로서,(소프트맥스)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# cost가 최소화 될수있게 경사하강법을 이용해 가장 낮은 cost를 찾는 옵티마이저
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
'''



