#!/usr/bin/env python3
"""
Module that builds, trains, and saves a neural network classifier
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network classifier
    
    Args:
        X_train (ndarray): Training input data
        Y_train (ndarray): Training labels
        X_valid (ndarray): Validation input data
        Y_valid (ndarray): Validation labels  
        layer_sizes (list): Number of nodes in each layer
        activations (list): Activation functions for each layer
        alpha (float): Learning rate
        iterations (int): Number of training iterations
        save_path (str): Path to save the model
        
    Returns:
        str: Path where the model was saved
    """
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            cost_train, accuracy_train = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            cost_valid, accuracy_valid = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})
            
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(accuracy_valid))
            
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
                
        save_path = saver.save(sess, save_path)
    return save_path
