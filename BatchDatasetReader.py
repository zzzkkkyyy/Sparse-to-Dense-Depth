"""
Code from https://github.com/shekkizh/FCN.tensorflow
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options = {}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        #self.images = np.array([self._transform(item['image']) for item in records_list])
        #self.annotations= np.array([np.expand_dims(self._transform(item['annotation']), axis = 3) for item in records_list])
        self.images = np.array([item['image'] for item in records_list])
        self.annotations= np.array([np.expand_dims(item['annotation'], axis = 3) for item in records_list])
        self.image_options = image_options
        
    def _transform(self, image):
        """
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            print(image_options)
            resize_width, resize_height = int(self.image_options["resize_width"]), int(self.image_options["resize_height"])
            resize_image = misc.imresize(image, [resize_width, resize_height], interp = 'nearest')
        else:
            resize_image = image
        """
        resize_width, resize_height = 192, 256
        resize_image = misc.imresize(image, [resize_width, resize_height], interp = 'nearest')
        return resize_image
    
    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset = 0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.images[start: end], self.annotations[start: end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size = [batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
