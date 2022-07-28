import tensorflow as tf


def huber_loss(X, Y):
    err = X - Y
    loss = tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)
    loss = tf.reduce_sum(loss)  # reduce sum is just summing of elements across dimensions in a vector (just addition)

    return loss


def mse_loss(X, Y):
    err = X - Y
    return tf.reduce_sum(tf.square(err))
