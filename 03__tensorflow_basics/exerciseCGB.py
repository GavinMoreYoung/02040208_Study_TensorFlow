import tensorflow as tf
a = tf.constant(6)
b = tf.constant(6)
c = tf.multiply(a, b)
sess = tf.Session()
d = tf.sin(float(sess.run(c)))
e = tf.divide(float(sess.run(b)), d)
e = sess.run(e)
sess.close()
print(e)

