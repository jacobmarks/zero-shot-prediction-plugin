"""Zero Shot Instance Segmentation.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import os
import sys


import fiftyone as fo
from fiftyone.core.models import Model
from fiftyone.core.utils import add_sys_path
import fiftyone.zoo as foz

def OwlViT_activator():
    return find_spec("transformers") is not None

def SAM_activator():
    return True

def OwlViT_SAM_activator():
    return OwlViT_activator() and SAM_activator()


INSTANCE_SEGMENTATION_MODELS = {
    "OwlViT + SAM ViT-B": {
        "activator": OwlViT_SAM_activator,
        "model": "N/A",
        "name": "OwlViT + SAM ViT-B",
        "segmentation_model_name": "segment-anything-vitb-torch",
    },
    "OwlViT + SAM ViT-H": {
        "activator": OwlViT_SAM_activator,
        "model": "N/A",
        "name": "OwlViT + SAM ViT-H",
        "segmentation_model_name": "segment-anything-vith-torch",
    },
    "OwlViT + SAM ViT-L": {
        "activator": OwlViT_SAM_activator,
        "model": "N/A",
        "name": "OwlViT + SAM ViT-L",
        "segmentation_model_name": "segment-anything-vitl-torch",
    },
}


def _get_model(model_name, config):
    return INSTANCE_SEGMENTATION_MODELS[model_name]["model"](config)


def run_two_step_instance_segmentation(dataset, model_name, label_field, categories):
    with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
        # pylint: disable=no-name-in-module,import-error
        from detection import run_zero_shot_detection
    
    detection_model_name = model_name.split(" + ")[0]
    segmentation_model_name = INSTANCE_SEGMENTATION_MODELS[model_name]["segmentation_model_name"]
    seg_model = foz.load_zoo_model(segmentation_model_name)
    run_zero_shot_detection(dataset, detection_model_name, label_field, categories)
    dataset.apply_model(seg_model, label_field=label_field, prompt_field=label_field)


def run_zero_shot_instance_segmentation(dataset, model_name, label_field, categories):
    if "SAM ViT" in model_name:
        run_two_step_instance_segmentation(dataset, model_name, label_field, categories)
    else:
        config = {"categories": categories}
        model = _get_model(model_name, config)
        dataset.apply_model(model, label_field=label_field)
