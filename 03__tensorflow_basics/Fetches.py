import tensorflow as tf

a = tf.constant(6)
b = tf.constant(6)
c = tf.multiply(a, b)
d = tf.add(a, b)
e = tf.subtract(d, c)
f = tf.add(c, d)
with tf.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs = sess.run(fetches)
print("outs = {}".format(outs))
print(type(outs[0]))
