from models import (base, cbam_base, cbam_resmob, 
                    bknet, refined_model, final, finalv2, 
                    inception_resnet, efficientnet,
                    mobilenetv1, mobilenetv2, mobilenetv3,
                    resnet, finalv2_conv, vgg16_cbam, google_net, vgg19)
from utils.checkpoint import restore
from utils.logger import Logger
from utils.helper import init_weights, count_parameters

nets = {
    'base': base.Model,
    'cbam_base': cbam_base.Model,
    'cbam_resmob': cbam_resmob.Model,
    'bknet': bknet.Model,
    'refined_model': refined_model.Model,
    'final': final.Model,
    'finalv2': finalv2.Model,
    'inception_resnet': inception_resnet.Model,
    'efficientnet': efficientnet.Model,
    'mobilenetv1': mobilenetv1.Model,
    'mobilenetv2': mobilenetv2.Model,
    'mobilenetv3': mobilenetv3.Model,
    'resnet': resnet.Model,
    'finalv2_conv': finalv2_conv.Model,
    'vgg16_cbam': vgg16_cbam.Model,
    'google_net': google_net.Model,
    'vgg19': vgg19.Model
}

def setup_network(network, in_channels, num_classes=7):
    
    print('model: ', network)
    net = nets[network](in_channels=in_channels, num_classes=num_classes)
    if network != 'inception_resnet' or network != 'efficientnet':
        net.apply(init_weights)
    print(f'total trainable parameters: {count_parameters(net)}')

    # Prepare logger
    logger = Logger()

    return logger, net