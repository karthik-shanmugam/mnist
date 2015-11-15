def convert(x):
    return (2 * x).astype(int)




import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs = convert(batch_xs)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: convert(mnist.test.images), y_: mnist.test.labels})
# karthik's messing around
import numpy
from PIL import Image
# test_image = tf.to_float(
#   tf.convert_to_tensor(
#       numpy.matrix(
#           numpy.random.rand(784))))
# result = sess.run(tf.matmul(test_image, W) + b)
# print result
# img = Image.open("test.png")
# arr = numpy.array(img)
# arr = (255 - arr) / 255.0
# #print(arr)
# arr = numpy.array(numpy.reshape(arr, (1, 784)))
# test_arr = tf.to_float(
#   tf.convert_to_tensor(
#           numpy.matrix(
#               arr)))
# t0 = arr
# result = sess.run(tf.matmul(test_arr, W) + b)
# print result

# img = Image.open("5.png")
# arr = numpy.array(img)
# arr = (255 - arr) / 255.0
# #print(arr)
# arr = numpy.array(numpy.reshape(arr, (1, 784)))
# test_arr = tf.to_float(
#   tf.convert_to_tensor(
#           numpy.matrix(
#               arr)))
# t1 = arr
# result = sess.run(tf.matmul(test_arr, W) + b)
# print result

# img = Image.open("6.png")
# arr = numpy.array(img)
# arr = (255 - arr) / 255.0
# #print(arr)
# arr = numpy.array(numpy.reshape(arr, (1, 784)))
# test_arr = tf.to_float(
#   tf.convert_to_tensor(
#           numpy.matrix(
#               arr)))
# t2 = arr
# result = sess.run(tf.matmul(test_arr, W) + b)
# print result

# combined = tf.to_float(
#       tf.convert_to_tensor(
#       numpy.matrix(
#           numpy.array(
#               [t0[0], t1[0], t2[0]]
#               ))))
# result = sess.run(tf.nn.softmax(tf.matmul(combined, W) + b))
# print result
# print [numpy.argmax(line) for line in result]


# a = (255 - 
#           numpy.array([
#               numpy.reshape(
#                   numpy.array(
#                       Image.open("%d1.png" % i)
#                       ),
#                   (1, 784))[0] 
#               for i in (0, 4)])
#           ) / 255.0
images = (255 - numpy.array([
                numpy.reshape(
                    numpy.array(
                        Image.open("%d.png" % i)
                        ),
                    (1, 784))[0] 
                for i in range(10)])
            ) / 128
samples = tf.to_float(
    tf.convert_to_tensor(
        images
    ))
def classify(image):
    show(image)
    image = tf.to_float(tf.convert_to_tensor(image.reshape(1, 784)))
    return sess.run(tf.nn.softmax(tf.matmul(image, W) + b))
result = sess.run(tf.nn.softmax(tf.matmul(samples, W) + b))
# for image in images:
#   print image.reshape(28,28)
print [numpy.argmax(line) for line in result]

def sample(i):
    tmp, ans = mnist.test.images[i], mnist.test.labels[i]
    tmp = convert(tmp)
    print numpy.argmax(classify(tmp))
def show(image):
    print image.reshape(28,28)
def test(i):
    print numpy.argmax(classify(images[i]))
def classify_image(filename):
    print numpy.argmax(classify((255 - numpy.array(Image.open(filename))) / 128))
# tx, ty = mnist.train.next_batch(5)

# result = sess.run(tf.nn.softmax(tf.matmul(tx, W) + b))
# print [numpy.argmax(line) for line in result]
# print [numpy.argmax(line) for line in ty]







