import threading
import time

import tensorflow as tf

sess = tf.InteractiveSession()
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100,dtypes=[tf.float32],shapes=())
enque = queue.enqueue(gen_random_normal)
def add():
    for i in range(10):
        sess.run(enque)
threads = [threading.Thread(target=add,args=()) for i in range(10)]
print(threads)

for t in threads:
    t.start()
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

x = queue.dequeue_many(10)
print(x.eval())
sess.run(queue.size())