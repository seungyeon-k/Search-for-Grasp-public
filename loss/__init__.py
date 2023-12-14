from loss.sq_loss import SuperquadricLoss
from .motion_prediction_loss import MotionPredictionLoss
from .segmentation_loss import SegmentationLoss
from .sq_loss import SuperquadricLoss


def get_loss(cfg_loss, *args, **kwargs):
    name = cfg_loss.type
    loss_instance = get_loss_instance(name)
    return loss_instance(**cfg_loss)

def get_loss_instance(name):
    try:
        return {
            'sq_loss': SuperquadricLoss,
            'segmentation_loss': SegmentationLoss,
            'motion_loss': MotionPredictionLoss,
        }[name]
    except:
        raise ("Loss {} not available".format(name))
