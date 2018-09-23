import tensorflow as tf


class Trainer:

    def __init__(self, sess, training_data, file_manager, eval_rate=16, batch_record_rate=100):
        self.sess = sess
        self.training_data = training_data
        self.file_manager = file_manager
        self.eval_rate = eval_rate
        self.batch_record_rate = batch_record_rate

    def start_training(self, network, epochs, learning_rate=10**(-5), step=0, max_steps=None, verbose=False):
        sess = self.sess
        with sess.graph.as_default():
            # get the input tensor of the network
            input_tf = network.input_tf
            # get the prediction tensor of the network
            pred_tf = network.pred_tf
            # get the training label tensor
            label_tf = self.training_data.get_label_tf()
            # create loss tensor based on prediction tensor and label tensor
            loss_tf = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_tf, logits=pred_tf, name="loss")
            # create summary of loss
            scalar_loss_tf = tf.reduce_mean(loss_tf, axis=0)
            scalar_loss_tf = tf.reduce_sum(scalar_loss_tf)
            loss_summary = tf.summary.scalar("sigmoid_loss", scalar_loss_tf)
            # create summary writers
            train_writer = tf.summary.FileWriter(self.file_manager.train_progress_path, sess.graph)
            validation_writer = tf.summary.FileWriter(self.file_manager.validation_progress_path, sess.graph)
            # create history of losses
            train_loss_history = list()
            validation_loss_history = list()
            # initialize the trainer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            trainer = optimizer.minimize(loss_tf)
            sess.run(tf.variables_initializer(optimizer.variables()))
            # create network saver
            saver = tf.train.Saver()
        if verbose:
            print("starting to train")
        for epoch in range(epochs):
            for (batch, labels) in self.training_data.get_batches():
                if verbose:
                    print("Got batch")
                step += 1
                _, loss_val, summary = sess.run([trainer, scalar_loss_tf, loss_summary],
                                                feed_dict={input_tf: batch, label_tf: labels})
                # record loss to history
                train_loss_history.append((step, loss_val))
                # write loss to summary writer
                train_writer.add_summary(summary, global_step=step)
                train_writer.flush()
                if verbose:
                    print("Step: %d, loss: %f" % (step, loss_val))

                if step % self.eval_rate == 0:
                    # perform an validation test on the network
                    (val_batch, val_labels) = self.training_data.get_val_batch()
                    loss_val, summary = sess.run([scalar_loss_tf, loss_summary],
                                                 feed_dict={input_tf: val_batch, label_tf: val_labels})
                    if verbose:
                        print("Step %d, validation loss: %f" % (step, loss_val))
                    # record validation loss to history
                    validation_loss_history.append((step, loss_val))
                    # write validation to summary writer
                    validation_writer.add_summary(summary, global_step=step)
                    validation_writer.flush()
                if step % self.batch_record_rate == 0:
                    # save the network
                    saver.save(sess, self.file_manager.network_save_path(step))
                if max_steps is not None and step >= max_steps:
                    # save the network
                    saver.save(sess, self.file_manager.network_save_path(step))
                    return train_loss_history, validation_loss_history
        return train_loss_history, validation_loss_history
