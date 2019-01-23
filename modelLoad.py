import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 로그출력 레벨설정
try:
    import cv2
except ImportError:
    pass
import tensorflow as tf
import matplotlib


cap = cv2.VideoCapture('img_data\\car3.mp4')
while(True):
    ret,frame=cap.read()
    #grayframe=cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("video",grayframe)


sess = tf.Session()
# create Model Network
saver = tf.train.import_meta_graph('model\\testModel.ckpt-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('model\\'))
graph = tf.get_default_graph()

# 1.자료형  2,데이터 크기에 맞출거면 None  3.입력 데이터크기
X = tf.placeholder(tf.float32,[None,6400])
Y = tf.placeholder(tf.float32,[None,2])
# layer 1
w1 = graph.get_tensor_by_name("w1:0")
b1 = graph.get_tensor_by_name("b1:0")
L1 = tf.nn.relu(tf.matmul(X, w1)+b1)
# layer 2
w2 = graph.get_tensor_by_name("w2:0")
b2 = graph.get_tensor_by_name("b2:0")
L2 = tf.nn.relu(tf.matmul(L1, w2)+b2)
# layer 3
w3 = graph.get_tensor_by_name("w3:0")
b3 = graph.get_tensor_by_name("b3:0")
model = tf.add(tf.matmul(L2,w3),b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
saver.restore(sess,'model\\testModel.ckpt')


prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis = 1)
# 모델의 예측 비 계산
#print('모델의 예측값', sess.run(prediction, feed_dict = {X: test_features}))
#print('      실제 값', sess.run(target, feed_dict={Y: test_labels}))

# 정확도 계산
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#print('accuracy: %.2f' % sess.run(accuracy*100, feed_dict = {X: test_features, Y: test_labels}))
