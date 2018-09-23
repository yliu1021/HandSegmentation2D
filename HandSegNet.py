from NetworkLayers import NetworkLayers as nn


class HandSegNet:

    def __init__(self):
        pass

    @staticmethod
    def inception_v8_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        prev_layer = nn.conv(prev_layer, "init_conv_1", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        for i in range(2):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_1")
        for i in range(2):
            prev_layer = nn.inception_module_base(prev_layer, 128, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_2")
        for i in range(4):
            prev_layer = nn.inception_module_simple(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        for i in range(4):
            prev_layer = nn.inception_module_simple(prev_layer, 256, "inception_4_%d" % i, trainable=trainable)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        out_tensor = nn.resize_images(out_tensor, original_shape)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=3, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v7_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        prev_layer = nn.conv(prev_layer, "init_conv_1", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        prev_layer = nn.conv(prev_layer, "init_conv_2", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_1")
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 128, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_2")
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 256, "inception_3_%d" % i, trainable=trainable)
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 512, "inception_4_%d" % i, trainable=trainable)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        out_tensor = nn.resize_images(out_tensor, original_shape)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=3, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v6_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        prev_layer = nn.conv(prev_layer, "init_conv_1", kernel_size=3, stride=1, out_chan=16, trainable=trainable)
        prev_layer = nn.conv(prev_layer, "init_conv_2", kernel_size=3, stride=1, out_chan=32, trainable=trainable)
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 32, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_1")
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 64, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.avg_pool(prev_layer, name="pool_2")
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        for i in range(2):
            prev_layer = nn.inception_module_simple(prev_layer, 512, "inception_4_%d" % i, trainable=trainable)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        out_tensor = nn.resize_images(out_tensor, original_shape)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=3, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v5_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 32, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_1")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_2")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 512, "inception_5_%d" % i, trainable=trainable)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=64, trainable=trainable)
        out_tensor = nn.resize_images(out_tensor, original_shape)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=3, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v4_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 32, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_1")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_2")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 256, "inception_4_%d" % i, trainable=trainable)
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 512, "inception_5_%d" % i, trainable=trainable)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=1, stride=1, out_chan=16, trainable=trainable)
        out_tensor = nn.resize_images(out_tensor, original_shape)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=3, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v3_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 32, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_1")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_2")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_3")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 256, "inception_4_%d" % i, trainable=trainable)
        prev_layer = nn.resize_images(prev_layer, original_shape)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=32, trainable=trainable)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=1, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_v2_segmentation(image_tensor, trainable=True):
        original_shape = image_tensor.get_shape().as_list()[1:3]
        prev_layer = image_tensor
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 32, "inception_1_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_1")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 64, "inception_2_%d" % i, trainable=trainable)
        prev_layer = nn.max_pool(prev_layer, name="pool_1")
        for i in range(1):
            prev_layer = nn.inception_module_base(prev_layer, 128, "inception_3_%d" % i, trainable=trainable)
        prev_layer = nn.resize_images(prev_layer, original_shape)
        out_tensor = nn.conv(prev_layer, "inter_final_conv", kernel_size=3, stride=1, out_chan=32, trainable=trainable)
        out_tensor = nn.conv(out_tensor, "final_conv", kernel_size=1, stride=1, out_chan=1, trainable=trainable)
        return out_tensor

    @staticmethod
    def inception_segmentation(image_tensor, trainable=True):
        aux_losses = list()
        prev_layer = image_tensor
        for i in range(6):
            prev_layer = nn.inception_module_base(prev_layer, 20, "inception%d" % i, trainable=trainable)
            if i == 2 or i == 4:
                tmp_out_tensor = nn.conv(prev_layer, "%d_final_conv" % i, kernel_size=1, stride=1,
                                         out_chan=1, trainable=trainable)
                aux_losses.append(tmp_out_tensor)
        out_tensor = nn.conv(prev_layer, "final_conv", kernel_size=1, stride=1, out_chan=1, trainable=trainable)
        return out_tensor, aux_losses

    @staticmethod
    def mobilenetv2_segmentation(image_tensor, trainable=True):
        # out_channels = [32] * 8 + [16] * 8
        out_channels = [64] * 3 + [128] * 3 + [256] * 3 + [512] * 3
        expansion_factors = [4] * 3 + [2] * 3 + [1] * 3 + [1] * 3
        initial_conv = nn.conv(image_tensor, "initial_conv", kernel_size=3, stride=1, out_chan=32, trainable=trainable)
        prev_layer = initial_conv
        for (i, (out_channel, expansion_factor)) in enumerate(zip(out_channels, expansion_factors)):
            prev_layer = nn.mobilenetv1_module_residual(prev_layer,
                                                        layer_name="mobilenet_residual_{}".format(i),
                                                        expansion_factor=expansion_factor, out_chan=out_channel,
                                                        trainable=trainable)
        out_tensor = nn.conv(prev_layer, "final_conv", kernel_size=1, stride=1, out_chan=1, trainable=trainable)
        return out_tensor
