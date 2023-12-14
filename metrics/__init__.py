from .segmentation_metric import SegementationAccuracy


def get_metric(metric_dict, **kwargs):
    name = metric_dict.pop("type")
    metric_instance = get_metric_instance(name)
    return metric_instance(**metric_dict)


def get_metric_instance(name):
    try:
        return {
            'segmentation': SegementationAccuracy,
        }[name]
    except:
        raise ("Metric {} not available".format(name))
