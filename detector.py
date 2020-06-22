import time

from torch.autograd import Variable

from util import *


class Detector:
    def __init__(self, model, im_batches, batch_size, inp_dim, confidence, nms_thresh, classes, number_of_classes,
                 cuda):
        self.model = model
        self.im_batches = im_batches
        self.batch_size = batch_size
        self.confidence = confidence
        self.inp_dim = inp_dim
        self.nms_thresh = nms_thresh
        self.classes = classes
        self.number_of_classes = number_of_classes
        self.cuda = cuda

    def detect(self, list_of_images, im_dim_list):
        write = 0
        for i, batch in enumerate(self.im_batches):
            # load the image
            start = time.time()
            if self.cuda:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = self.model(Variable(batch), self.cuda)

            prediction = write_results(prediction, self.confidence, self.number_of_classes, nms_conf=self.nms_thresh)

            end = time.time()

            if type(prediction) == int:

                for im_num, image in enumerate(
                        list_of_images[i * self.batch_size: min((i + 1) * self.batch_size, len(list_of_images))]):
                    im_id = i * self.batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                         (end - start) / self.batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------------------------------------------------")
                continue

            prediction[:, 0] += i * self.batch_size  # transform the atribute from index in batch to index in imlist

            if not write:  # If we have't initialised output
                output = prediction
                write = 1
            else:
                output = torch.cat((output, prediction))

            for im_num, image in enumerate(
                    list_of_images[i * self.batch_size: min((i + 1) * self.batch_size, len(list_of_images))]):
                im_id = i * self.batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1],
                                                                     (end - start) / self.batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")

            if self.cuda:
                torch.cuda.synchronize()
        try:
            output
        except NameError:
            print("No detections were made")
            exit()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
        return output
