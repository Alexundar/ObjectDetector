from __future__ import division

import time

from arguments_parser import ArgumentsParser
from darknet import Darknet
from detector import Detector
from image_manager import ImageManager
from util import *

CUDA = torch.cuda.is_available()
NUMBER_OF_CLASSES = 80
CLASSES = load_classes("data/coco.names")


def main():
    arguments_parser = ArgumentsParser()
    args = arguments_parser.parse_arguments()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
    load_batch = time.time()
    image_manager = ImageManager()
    loaded_images, list_of_images = image_manager.read_images(images)
    im_batches = list(map(prep_image, loaded_images, [inp_dim for x in range(len(list_of_images))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_images]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(list_of_images) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    start_det_loop = time.time()
    detector = Detector(model, im_batches, batch_size, inp_dim, confidence, nms_thresh, CLASSES, NUMBER_OF_CLASSES,
                        CUDA)
    output = detector.detect(list_of_images, im_dim_list)

    output_recast = time.time()
    class_load = time.time()

    draw = time.time()

    det_images = list(map(lambda x: image_manager.draw_bounding_boxes(x, loaded_images, CLASSES), output))
    det_names = list(map(lambda x: "{}/det_{}.jpg".format(args.det, x), list(range(len(list_of_images)))))
    image_manager.write_images(det_names, det_images)

    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print(
        "{:25s}: {:2.3f}".format("Detection (" + str(len(list_of_images)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(list_of_images)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
