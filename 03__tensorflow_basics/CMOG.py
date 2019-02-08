import tensorflow as tf
print(tf.get_default_graph())
g = tf.Graph()
print(g)
a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())
print("------------------------")
g1 = tf.get_default_graph()
g2 = tf.Graph()
print(g1 is tf.get_default_graph())
with g2.as_default():
    print(g1 is tf.get_default_graph())
print(g1 is tf.get_default_graph())
