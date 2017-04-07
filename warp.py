import numpy as np
import scipy.linalg
import tensorflow as tf


# fit a warp between two sets of points
def fit(Xsrc, Xdst, warpType):
    ptsN = len(Xsrc)
    X, Y, U, V, O, I = Xsrc[:, 0], Xsrc[:, 1], Xdst[:, 0], Xdst[:, 1], np.zeros([ptsN]), np.ones([ptsN])
    if warpType == "similarity":
        A = np.concatenate((np.stack([X, -Y, I, O], axis=1),
                            np.stack([Y, X, O, I], axis=1)), axis=0)
        b = np.concatenate((U, V), axis=0)
        p = scipy.linalg.lstsq(A, b)[0].squeeze()
        warpMtrx = np.array([[p[0], -p[1], p[2]], [p[1], p[0], p[3]], [0, 0, 1]], dtype=np.float32)
    elif warpType == "affine":
        A = np.concatenate((np.stack([X, Y, I, O, O, O], axis=1),
                            np.stack([O, O, O, X, Y, I], axis=1)), axis=0)
        b = np.concatenate((U, V), axis=0)
        p = scipy.linalg.lstsq(A, b)[0].squeeze()
        warpMtrx = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]], dtype=np.float32)
    return warpMtrx


# compute composition of batch of warp parameters
def compose(pBatch, dpBatch, params, name=None):
    with tf.name_scope("compose"):
        pMtrxBatch = vec2mtrxBatch(pBatch, params)
        dpMtrxBatch = vec2mtrxBatch(dpBatch, params)
        pMtrxNewBatch = tf.matmul(dpMtrxBatch, pMtrxBatch)
        pMtrxNewBatch = pMtrxNewBatch / tf.slice(pMtrxNewBatch, [0, 2, 2], [-1, 1, 1])
        pNewBatch = mtrx2vecBatch(pMtrxNewBatch, params)
        pNewBatch = tf.identity(pNewBatch, name=name)
    return pNewBatch


# compute inverse of warp parameters
def inverse(pBatch, params, name=None):
    with tf.name_scope("inverse"):
        pMtrxBatch = vec2mtrxBatch(pBatch, params)
        pInvMtrxBatch = tf.matrix_inverse(pMtrxBatch)
        pInvBatch = mtrx2vecBatch(pInvMtrxBatch, params)
        pInvBatch = tf.identity(pInvBatch, name=name)
    return pInvBatch


# convert batch of warp parameters to matrix
def vec2mtrxBatch(pBatch, params):
    with tf.name_scope("vec2mtrx"):
        batchSize = tf.shape(pBatch)[0]
        O = tf.zeros([batchSize])
        I = tf.ones([batchSize])
        if params.warpType == "translation":
            tx, ty = tf.unstack(pBatch, axis=1)
            pMtrxBatch = tf.transpose(tf.stack([[I, O, tx],
                                                [O, I, ty],
                                                [O, O, I]]), perm=[2, 0, 1])
        elif params.warpType == "similarity":
            pc, ps, tx, ty = tf.unstack(pBatch, axis=1)
            pMtrxBatch = tf.transpose(tf.stack([[I + pc, -ps, tx],
                                                [ps, I + pc, ty],
                                                [O, O, I]]), perm=[2, 0, 1])
        elif params.warpType == "affine":
            p1, p2, p3, p4, p5, p6 = tf.unstack(pBatch, axis=1)
            pMtrxBatch = tf.transpose(tf.stack([[I + p1, p2, p3],
                                                [p4, I + p5, p6],
                                                [O, O, I]]), perm=[2, 0, 1])
        elif params.warpType == "homography":
            p1, p2, p3, p4, p5, p6, p7, p8 = tf.unstack(pBatch, axis=1)
            pMtrxBatch = tf.transpose(tf.stack([[I + p1, p2, p3],
                                                [p4, I + p5, p6],
                                                [p7, p8, I]]), perm=[2, 0, 1])
    return pMtrxBatch


# convert batch of warp matrix to parameters
def mtrx2vecBatch(pMtrxBatch, params):
    with tf.name_scope("mtrx2vec"):
        if params.warpType == "translation":
            [row0, row1, row2] = tf.unstack(pMtrxBatch, axis=1)
            [e00, e01, e02] = tf.unstack(row0, axis=1)
            [e10, e11, e12] = tf.unstack(row1, axis=1)
            [e20, e21, e22] = tf.unstack(row2, axis=1)
            pBatch = tf.stack([e02, e12], axis=1)
        elif params.warpType == "similarity":
            [row0, row1, row2] = tf.unstack(pMtrxBatch, axis=1)
            [e00, e01, e02] = tf.unstack(row0, axis=1)
            [e10, e11, e12] = tf.unstack(row1, axis=1)
            [e20, e21, e22] = tf.unstack(row2, axis=1)
            pBatch = tf.stack([e00 - 1, e10, e02, e12], axis=1)
        elif params.warpType == "affine":
            [row0, row1, row2] = tf.unstack(pMtrxBatch, axis=1)
            [e00, e01, e02] = tf.unstack(row0, axis=1)
            [e10, e11, e12] = tf.unstack(row1, axis=1)
            [e20, e21, e22] = tf.unstack(row2, axis=1)
            pBatch = tf.stack([e00 - 1, e01, e02, e10, e11 - 1, e12], axis=1)
        elif params.warpType == "homography":
            pMtrxBatch = pMtrxBatch / tf.slice(pMtrxBatch, [0, 2, 2], [-1, 1, 1])
            [row0, row1, row2] = tf.unstack(pMtrxBatch, axis=1)
            [e00, e01, e02] = tf.unstack(row0, axis=1)
            [e10, e11, e12] = tf.unstack(row1, axis=1)
            [e20, e21, e22] = tf.unstack(row2, axis=1)
            pBatch = tf.stack([e00 - 1, e01, e02, e10, e11 - 1, e12, e20, e21], axis=1)
    return pBatch
