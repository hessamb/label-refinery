"""Utility functions to construct a model."""

import torch
from torch import nn

from extensions import data_parallel
from extensions import model_refinery_wrapper
from extensions import refinery_loss
from models import alexnet
from models import resnet50


MODEL_NAME_MAP = {
    'AlexNet': alexnet.AlexNet,
    'ResNet50': resnet50.ResNet50,
}


def _create_single_cpu_model(model_name, state_file=None):
    if model_name not in MODEL_NAME_MAP:
        raise ValueError("Model {} is invalid. Pick from {}.".format(
            model_name, sorted(MODEL_NAME_MAP.keys())))
    model_class = MODEL_NAME_MAP[model_name]
    model = model_class()
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
    return model


def create_model(model_name, model_state_file=None, gpus=[], label_refinery=None,
                 label_refinery_state_file=None):
    model = _create_single_cpu_model(model_name, model_state_file)
    if label_refinery is not None:
        assert label_refinery_state_file is not None, "Refinery state is None."
        label_refinery = _create_single_cpu_model(
            label_refinery, label_refinery_state_file)
        model = model_refinery_wrapper.ModelRefineryWrapper(model, label_refinery)
        loss = refinery_loss.RefineryLoss()
    else:
        loss = nn.CrossEntropyLoss()

    if len(gpus) > 0:
        model = model.cuda()
        loss = loss.cuda()
    if len(gpus) > 1:
        model = data_parallel.DataParallel(model, device_ids=gpus)
    return model, loss
