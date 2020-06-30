import argparse

from arguments_parser.abstract_arguments_parser import AbstractArgumentsParser


class ArgumentsParser(AbstractArgumentsParser):

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
        parser.add_argument("--images", dest='images',
                            help="Image / Directory containing images to perform detection upon",
                            default="bot/photos", type=str)
        parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                            default="bot/det_photos", type=str)
        parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
        parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions",
                            default=0.5)
        parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
        parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
        parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
        parser.add_argument("--reso", dest='reso', help=
        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416",
                            type=str)

        return parser.parse_args()
