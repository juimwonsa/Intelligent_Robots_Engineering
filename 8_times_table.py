import tensorflow as tf
# input training data X
X=[[1], [2], [3], [4], [5], [6], [7], [8], [9]]              # dimension 9X1

# output target data Y
Y=[[8], [16], [24], [32], [40], [48], [56], [64], [72]]            # dimension 9X1

# placeholder for input and output
x= tf.placeholder(tf.float32, shape=[9,1])  # 입력학습 데이터 X가 저장될 공간
y= tf.placeholder(tf.float32, shape=[9,1])  # 출력학습 데이터 Y가 저장될 공간

# Weight matrix W1
W1 = tf.Variable([[1.0]],shape=[1,1])

#bias
B1 = tf.Variable([0.0], shape=[1])

# Hidden layer and output layer
output = tf.nn.relu(tf.matmul(x,W1)+B1)

# error estimation
e = tf.reduce_mean(tf.squared_difference(y,output))   # 입력된 값에 대한 신경망 출력과 target 간의 차이를 계산함
train = tf.train.AdamOptimizer(0.001).minimize(e)

init = tf.global_variables_initializer()     # 사용된 변수의 초기화(Weight 값의 초기화; W1)
sess = tf.Session()
sess.run(init)                              #  Weight W1 초기화 수행  

for i in range(100001):
    error = sess.run(train, feed_dict={x:X, y:Y})    
    if i % 10000 == 0:
        print('\nEpoch: ' + str(i))
        print('\nError: ' + str(sess.run(e,feed_dict={x:X, y:Y})))
        for e1 in sess.run(output, feed_dict={x:X, y:Y}):
            print('     ', e1)
            
print(sess.run(output,feed_dict={x:[[8],[8],[7.8],[6],[3.3],[2],[1],[6.2],[9.7]]}))
sess.close()

print("complete")
            







