from __future__ import division
import random
import pickle
import cv2
import numpy as np
import imutils
import os
from multiprocessing import Process, Queue
from Queue import Empty
import time


class Pipeline:

    def __init__(self, batch_size, num_processes, cache_location_base, color_files, mask_files,
                 augment=True, shuffle=True, update_callback=None):
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.augment = augment
        self.update_callback = update_callback

        self.cache_location_base = cache_location_base
        self.color_files = color_files
        self.mask_files = mask_files
        self.filenames = os.listdir(self.color_files)
        if shuffle:
            self.filenames = random.shuffle(self.filenames)

        # try to make each process fetch the same number of batches
        self.batch_size_per_process = batch_size // num_processes
        self.leftover = batch_size % num_processes

    def __iter__(self):
        return self

    def next(self):
        inputs = list()
        outputs = list()
        processes = list()
        input_output_pair_queues = list()
        files_per_process = list()
        for _ in range(self.num_processes):
            if self.leftover > 0:
                # if we must, make some processes fetch one more batch
                self.leftover -= 1
                batch_size = self.batch_size_per_process + 1
            else:
                batch_size = self.batch_size_per_process
            files = self.filenames[:batch_size]
            del self.filenames[:batch_size]
            files_per_process.append(files)
        for process_id in range(self.num_processes):
            # create the processes
            input_output_pair_queue = Queue()
            files = files_per_process[process_id]
            process = Process(target=self.async_get_batch, name="fetch_process_%d" % process_id,
                              args=(input_output_pair_queue, files))
            process.start()
            processes.append(process)
            input_output_pair_queues.append(input_output_pair_queue)
        while True:
            # flush all input/output pairs into inputs and outputs list
            for input_output_pair_queue in input_output_pair_queues:
                while True:
                    try:
                        input_im, output_im = input_output_pair_queue.get_nowait()
                        inputs.append(input_im)
                        outputs.append(output_im)
                    except Empty:
                        break
            # if we need to update the caller on the amount already fetched
            if self.update_callback is not None:
                fetched_amount = len(inputs)
                self.update_callback(fetched_amount)
            # check that there are still processes running
            still_running = False
            for p in processes:
                still_running = still_running or p.is_alive()
            if not still_running:
                break
            else:
                # wait for more to be fetched
                time.sleep(0.25)
        if self.update_callback is not None:
            self.update_callback(None)
        return inputs, outputs

    def async_get_batch(self, input_output_pair_queue, filenames):
        for filename in filenames:
            (index, ext) = filename.split(".")
            if ext != 'png':
                continue

            color_file = self.color_files + filename
            mask_file = self.mask_files + filename

            input_im, output = self.get_input_output(color_file, mask_file)
            input_output_pair_queue.put((input_im, output))

    def get_input_output(self, i_color_file, i_mask_file):
        file_name = (i_color_file.split("/")[-1]).split(".")[-2]
        is_eval = i_color_file.split("/")[-3]
        cache_location = self.cache_location_base + is_eval + "/"
        cached_images = os.listdir(cache_location)
        image_cache_location = cache_location + file_name + "/"
        if file_name in cached_images:
            num_cache = len(os.listdir(image_cache_location))
            if num_cache == 2:
                with open(image_cache_location + "input.pickle", "rb") as handle:
                    input_image = pickle.load(handle)
                with open(image_cache_location + "mask.pickle", "rb") as handle:
                    output_image = pickle.load(handle)
                input_image, output_image = augment_image(input_image, output_image, just_crop=(not self.augment))

                # with open(image_cache_location + "1_input.pickle", "wb") as handle:
                #     pickle.dump(input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # with open(image_cache_location + "1_mask.pickle", "wb") as handle:
                #     pickle.dump(output_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                if num_cache < 2:
                    print(image_cache_location)
                index = random.randint(1, num_cache / 2 - 1)
                with open(image_cache_location + "%d_input.pickle" % index, "rb") as handle:
                    input_image = pickle.load(handle)
                with open(image_cache_location + "%d_mask.pickle" % index, "rb") as handle:
                    output_image = pickle.load(handle)
                # if num_cache < 30:
                #     with open(image_cache_location + "input.pickle", "rb") as handle:
                #         orig_input_image = pickle.load(handle)
                #     with open(image_cache_location + "mask.pickle", "rb") as handle:
                #         orig_output_image = pickle.load(handle)
                    # orig_input_image, orig_output_image = augment_image(orig_input_image, orig_output_image,
                    #                                                     just_crop=(not self.augment))
                    # with open(image_cache_location + "%d_input.pickle" % (num_cache / 2), "wb") as handle:
                    #     pickle.dump(orig_input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # with open(image_cache_location + "%d_mask.pickle" % (num_cache / 2), "wb") as handle:
                    #     pickle.dump(orig_output_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            output_image = cv2.imread(i_mask_file, 0)
            shape = output_image.shape
            for y in range(shape[0]):
                for x in range(shape[1]):
                    v = output_image[y][x]
                    if 2 <= v <= 17:
                        output_image[y][x] = 1
                    elif 18 <= v <= 33:
                        output_image[y][x] = 1
                    else:
                        output_image[y][x] = 0
            output_image = cv2.resize(output_image.astype('float32'), (0, 0), fx=0.5, fy=0.5)

            input_image = cv2.imread(i_color_file).astype('float32')
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)

            if not os.path.exists(image_cache_location):
                os.mkdir(image_cache_location)
            with open(image_cache_location + "input.pickle", "wb") as handle:
                pickle.dump(input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(image_cache_location + "mask.pickle", "wb") as handle:
                pickle.dump(output_image, handle, protocol=pickle.HIGHEST_PROTOCOL)

            input_image, output_image = augment_image(input_image, output_image, just_crop=(not self.augment))

            # with open(image_cache_location + "1_input.pickle", "wb") as handle:
            #     pickle.dump(input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(image_cache_location + "1_mask.pickle", "wb") as handle:
            #     pickle.dump(output_image, handle, protocol=pickle.HIGHEST_PROTOCOL)

        input_image = input_image / 255.0 - 0.5
        output_image = np.expand_dims(output_image, 2)

        return input_image, output_image


def with_chance(prob, aug_function, image, mask, *args, **kwargs):
    if random.uniform(0, 1) <= prob:
        return aug_function(image, mask, *args, **kwargs)
    else:
        return image, mask


def flip_vertically(image, mask):
    cv2.flip(image, 0, dst=image)
    cv2.flip(mask, 0, dst=mask)
    return image, mask


def flip_horizontally(image, mask):
    cv2.flip(image, 1, dst=image)
    cv2.flip(mask, 1, dst=mask)
    return image, mask


def random_rotate(image, mask):
    angle = random.uniform(-30, 30)
    image = imutils.rotate_bound(image, angle)
    mask = imutils.rotate_bound(mask, angle)
    return image, mask


def random_crop(image, mask, size=(128.0, 72.0)):
    dimensions = mask.shape
    max_x, max_y = 0, 0
    min_x, min_y = dimensions[1], dimensions[0]
    for y in range(dimensions[0]):
        for x in range(dimensions[1]):
            if mask[y][x] == 1:
                max_y = max(y, max_y)
                max_x = max(x, max_x)
                min_y = min(y, min_y)
                min_x = min(x, min_x)
    width = max_x - min_x
    height = max_y - min_y
    if width*size[1] > height*size[0]:
        #   width/height > crop_width/crop_height
        height = width*size[1]/size[0]
        max_height = min(dimensions[0], dimensions[1]/size[0]*size[1])
        if height > max_height:
            height = max_height
        else:
            scale = random.uniform(1, (float(max_height) / float(height)) ** 2)
            height = height * scale**(1.0/2.0)
        width = height * size[0] / size[1]
    else:
        width = height*size[0]/size[1]
        max_width = min(dimensions[1], dimensions[0]/size[1]*size[0])
        if width > max_width:
            width = max_width
        else:
            scale = random.uniform(1, (float(max_width) / float(width)) ** 2)
            width = width * scale**(1.0/2.0)
        height = width * size[1] / size[0]

    center_x, center_y = (min_x+max_x)/2, (min_y+max_y)/2

    crop_max_x = center_x + width/2
    crop_max_y = center_y + height/2
    crop_min_x = center_x - width/2
    crop_min_y = center_y - height/2

    if crop_max_x > dimensions[1]:
        diff = crop_max_x - dimensions[1]
        crop_max_x -= diff
        crop_min_x -= diff
    elif crop_min_x < 0:
        diff = -crop_min_x
        crop_min_x += diff
        crop_max_x += diff
    if crop_max_y > dimensions[0]:
        diff = crop_max_y - dimensions[0]
        crop_max_y -= diff
        crop_min_y -= diff
    elif crop_min_y < 0:
        diff = -crop_min_y
        crop_min_y += diff
        crop_max_y += diff

    crop_max_x = int(crop_max_x)
    crop_max_y = int(crop_max_y)
    crop_min_x = int(crop_min_x)
    crop_min_y = int(crop_min_y)

    x_shift_max = dimensions[1] - crop_max_x
    x_shift_min = -crop_min_x
    x_shift = int(random.uniform(x_shift_min, x_shift_max))
    crop_min_x += x_shift
    crop_max_x += x_shift

    y_shift_max = dimensions[0] - crop_max_y
    y_shift_min = -crop_min_y
    y_shift = int(random.uniform(y_shift_min, y_shift_max))
    crop_min_y += y_shift
    crop_max_y += y_shift

    image = image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
    mask = mask[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
    image = cv2.resize(image, (128, 72))
    mask = cv2.resize(mask, (128, 72))
    return image, mask


def hue_augmentation(image, mask):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue_adjustment = random.uniform(-1, 1)
    dimensions = hsv_image.shape
    for y in range(dimensions[0]):
        for x in range(dimensions[1]):
            hsv_image[y][x][0] = (hsv_image[y][x][0] + hue_adjustment) % 180
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return rgb_image, mask


def mask_blur(image, mask):
    mask = cv2.blur(mask, (7, 7))
    mask = cv2.blur(mask, (5, 5))
    mask = cv2.blur(mask, (5, 5))
    mask = cv2.blur(mask, (5, 5))
    return image, mask


def augment_image(image, mask, just_crop=False):
    if just_crop:
        return random_crop(image, mask)
    rotated_image, rotated_mask = with_chance(0.2, random_rotate, image, mask)
    cropped_image, cropped_mask = random_crop(rotated_image, rotated_mask)
    # cropped_image, cropped_mask = rotated_image, rotated_mask
    flipped_image, flipped_mask = with_chance(0.2, flip_horizontally, cropped_image, cropped_mask)
    flipped_image, flipped_mask = with_chance(-0.2, flip_vertically, flipped_image, flipped_mask)
    hue_adjusted_image, hue_adjusted_mask = with_chance(-0.1, hue_augmentation, flipped_image, flipped_mask)
    mask_blurred_images = mask_blur(hue_adjusted_image, hue_adjusted_mask)
    return mask_blurred_images
