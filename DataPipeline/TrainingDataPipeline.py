import tensorflow as tf
import HandSegDataPipeline


class TrainingDataPipeline:

    def __init__(self, file_manager, output_size, batch_size, num_processes, augment=True,
                 shuffle=True, update_callback=None):
        self.file_manager = file_manager
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.augment = augment
        self.shuffle = shuffle
        self.update_callback = update_callback

    def get_label_tf(self):
        label_tf = tf.placeholder(tf.float32, shape=[None]+self.output_size+[1], name="label")
        return label_tf

    def get_batches(self):
        return HandSegDataPipeline.Pipeline(self.batch_size, self.num_processes,
                                            self.file_manager.parameters["cache_location_base"],
                                            self.file_manager.parameters["color_files"],
                                            self.file_manager.parameters["mask_files"],
                                            self.augment, self.shuffle, self.update_callback)

    def get_val_batch(self):
        pipeline = HandSegDataPipeline.Pipeline(self.batch_size, self.num_processes,
                                                self.file_manager.parameters["cache_location_base"],
                                                self.file_manager.parameters["eval_color_files"],
                                                self.file_manager.parameters["eval_mask_files"],
                                                self.augment, self.shuffle, self.update_callback)
        pipeline_iter = iter(pipeline)
        return pipeline_iter.next()
