import tensorflow as tf
a = tf.constant(6)
b = tf.constant(6)
c = tf.multiply(a, b)
d = tf.add(a, b)
e = tf.subtract(d, c)
f = tf.add(c, d)
g = tf.divide(f, e)
with tf.Session() as sess:
    ans = sess.run(g)
print(ans)

