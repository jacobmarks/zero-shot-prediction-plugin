"""Zero Shot Instance Segmentation.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import os


from fiftyone.core.utils import add_sys_path
import fiftyone.zoo as foz

SAM_ARCHS = ("ViT-B", "ViT-H", "ViT-L")
SAM_MODELS = [("", SA) for SA in SAM_ARCHS]

# def OwlViT_activator():
#     return find_spec("transformers") is not None


def SAM_activator():
    return True


def build_instance_segmentation_models_dict():
    sms = {}

    if SAM_activator():
        sms["SAM"] = {
            "activator": SAM_activator,
            "model": "N/A",
            "name": "SAM",
            "submodels": SAM_MODELS,
        }

    return sms


INSTANCE_SEGMENTATION_MODELS = build_instance_segmentation_models_dict()


# def OwlViT_SAM_activator():
#     return OwlViT_activator() and SAM_activator()


# INSTANCE_SEGMENTATION_MODELS = {
#     "OwlViT + SAM ViT-B": {
#         "activator": OwlViT_SAM_activator,
#         "model": "N/A",
#         "name": "OwlViT + SAM ViT-B",
#         "segmentation_model_name": "segment-anything-vitb-torch",
#     },
#     "OwlViT + SAM ViT-H": {
#         "activator": OwlViT_SAM_activator,
#         "model": "N/A",
#         "name": "OwlViT + SAM ViT-H",
#         "segmentation_model_name": "segment-anything-vith-torch",
#     },
#     "OwlViT + SAM ViT-L": {
#         "activator": OwlViT_SAM_activator,
#         "model": "N/A",
#         "name": "OwlViT + SAM ViT-L",
#         "segmentation_model_name": "segment-anything-vitl-torch",
#     },
# }


# def _get_model(model_name, config):
#     return INSTANCE_SEGMENTATION_MODELS[model_name]["model"](config)


# def run_two_step_instance_segmentation(
#     dataset, model_name, label_field, categories
# ):
#     with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
#         # pylint: disable=no-name-in-module,import-error
#         from detection import run_zero_shot_detection

#     detection_model_name = model_name.split(" + ")[0]
#     segmentation_model_name = INSTANCE_SEGMENTATION_MODELS[model_name][
#         "segmentation_model_name"
#     ]
#     seg_model = foz.load_zoo_model(segmentation_model_name)
#     run_zero_shot_detection(
#         dataset, detection_model_name, label_field, categories
#     )
#     dataset.apply_model(
#         seg_model, label_field=label_field, prompt_field=label_field
#     )


def _get_segmentation_model(architecture):
    zoo_model_name = (
        "segment-anything-" + architecture.lower().replace("-", "") + "-torch"
    )
    return foz.load_zoo_model(zoo_model_name)


def run_zero_shot_instance_segmentation(
    dataset,
    model_name,
    label_field,
    categories,
    pretrained=None,
    architecture=None,
    **kwargs
):
    with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
        # pylint: disable=no-name-in-module,import-error
        from detection import run_zero_shot_detection

    det_model_name, _ = model_name.split(" + ")
    det_pretrained, _ = pretrained.split(" + ")
    if det_pretrained == "":
        det_pretrained = None
    _, seg_architecture = architecture.split(" + ")

    run_zero_shot_detection(
        dataset,
        det_model_name,
        label_field,
        categories,
        pretrained=det_pretrained,
    )

    seg_model = _get_segmentation_model(seg_architecture)
    dataset.apply_model(
        seg_model, label_field=label_field, prompt_field=label_field
    )

    # if "SAM ViT" in model_name:
    #     run_two_step_instance_segmentation(
    #         dataset, model_name, label_field, categories
    #     )
    # else:
    #     config = {"categories": categories}
    #     model = _get_model(model_name, config)
    #     dataset.apply_model(model, label_field=label_field)
