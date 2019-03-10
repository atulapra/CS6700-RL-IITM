import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def customOps(n):
    # Part 1
    B = tf.reverse(A,[1])
    C = tf.linalg.band_part(B,-1,0)
    D = tf.linalg.band_part(B,0,0)
    E = tf.subtract(C,D)
    F = tf.transpose(E)
    G = tf.reverse(F,[0])
    H = tf.subtract(B,E)
    I = tf.reverse(H,[1])
    J = tf.add(I,G)

    # Part 2
    m = tf.reduce_max(J,axis=1)

    # Part 3
    K = tf.reshape(tf.tile(m,[n]),[n,n])
    L = tf.zeros_like(J)
    M = tf.reverse(K,[1])
    N = tf.linalg.band_part(M,0,-1)
    P = tf.reverse(N,[1])            # to be softmaxed
    Q = tf.zeros_like(J)
    Qlog = tf.log(Q)
    Qlrt = tf.linalg.band_part(Qlog,0,-1)
    Qrev = tf.reverse(Qlrt,[1])
    Qdiff = tf.subtract(Qlog,Qrev)
    Qf = tf.where(tf.is_nan(Qdiff), tf.zeros_like(Qdiff), Qdiff)
    R = tf.nn.softmax(P + Qf)

    # Part 4
    S = tf.reduce_sum(R,1)

    # Part 5
    T = tf.reduce_sum(R,0)

    # Part 6
    U = tf.concat([S,T],axis=0)
    V = tf.nn.softmax(U)

    # Part 7
    W = tf.argmax(V)
    X = tf.cast(W,tf.float32)

    # Part 8
    comp = tf.constant(n/3)
    finalVal = tf.cond(X > comp, lambda: tf.norm(S-T), lambda: tf.norm(S+T))

    return finalVal

if __name__ == '__main__':
    mat = np.asarray([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    n = mat.shape[0]
    A = tf.placeholder(tf.float32,shape=(n,n))
    finalVal = customOps(n)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    outVal = sess.run(finalVal, feed_dict={A: mat})
    print(outVal)
    sess.close()