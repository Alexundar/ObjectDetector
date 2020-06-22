import torch.nn as nn

from detection_layer import DetectionLayer
from empty_layer import EmptyLayer


class ModuleCreator:
    def __init__(self, blocks):
        self.blocks = blocks

    def create_modules(self):
        net_info = self.blocks[0]  # Captures the information about the input and pre-processing
        module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []

        for index, x in enumerate(self.blocks[1:]):
            module = nn.Sequential()

            # check the type of block
            # create a new module for the block
            # append to module_list

            # If it's a convolutional layer
            if (x["type"] == "convolutional"):
                # Get the info about the layer
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True

                filters = int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                stride = int(x["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                # Add the convolutional layer
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)

                # Add the Batch Norm Layer
                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)

                # Check the activation.
                # It is either Linear or a Leaky ReLU for YOLO
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module("leaky_{0}".format(index), activn)

                # If it's an upsampling layer
                # We use Bilinear2dUpsampling
            elif (x["type"] == "upsample"):
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor=2, mode="nearest")
                module.add_module("upsample_{}".format(index), upsample)

            # If it is a route layer
            elif (x["type"] == "route"):
                x["layers"] = x["layers"].split(',')
                # Start  of a route
                start = int(x["layers"][0])
                # end, if there exists one.
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0
                # Positive anotation
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]

            # shortcut corresponds to skip connection
            elif x["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)

            # Yolo is the detection layer
            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)

            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

        return (net_info, module_list)
