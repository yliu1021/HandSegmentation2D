import tensorflow as tf
import numpy as np
import time
import sys
from NeuralNetwork import Network
from Trainer import Trainer
from DataPipeline.TrainingDataPipeline import TrainingDataPipeline
from FileManager import FileManager
from Genotype import Ribosome


class TrainingTask:

    def __init__(self, dna, parameters, base_path, name):
        # set up training data
        file_manager = FileManager(parameters, "train_progress/train",
                                   "train_progress/validation", base_path, name)
        pipeline = TrainingDataPipeline(file_manager, [72, 128], batch_size=32, num_processes=8, augment=True,
                                        shuffle=False, update_callback=None)
        self.pipeline = pipeline
        self.file_manager = file_manager
        self.name = name

        graph = tf.Graph()
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config, graph=graph)
        self.sess = sess

        layers = Ribosome.get_layers(dna.codons)
        dna.save_file(file_manager.dna_path)
        self.net = Network(sess, layers, trainable=True)
        self.net.initialize()

        self.trainer = Trainer(sess, pipeline, file_manager, eval_rate=10, batch_record_rate=100)

    def num_parameters(self):
        with self.sess.graph.as_default():
            # https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
            return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    def flops(self):
        flops = tf.profiler.profile(self.sess.graph,
                                    options=tf.profiler.ProfileOptionBuilder.float_operation(),
                                    op_log=sys.stdout)
        return int(flops.total_float_ops)

    def pass_through_time(self, passes=4):
        with self.sess.graph.as_default():
            input_tf = self.net.input_tf
            pred_tf = self.net.pred_tf
            (batch, _) = self.pipeline.get_val_batch()
            passes = min(len(batch), passes)
            total_time = 0

            for input_im in batch[:passes]:
                start = time.time()
                self.sess.run(pred_tf, feed_dict={input_tf: [input_im]})
                end = time.time()
                total_time += end - start
        return total_time / passes

    def train(self, max_steps, step=0, verbose=False):
        train_loss_history, validation_loss_history = self.trainer.start_training(self.net, epochs=4,
                                                                                  learning_rate=10 ** (-5), step=step,
                                                                                  max_steps=max_steps, verbose=verbose)
        return train_loss_history, validation_loss_history
