"""Zero Shot Detection.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import pkg_resources

from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.core.models import Model

YOLO_WORLD_PRETRAINS = (
    "yolov8s-world",
    "yolov8s-worldv2",
    "yolov8m-world",
    "yolov8m-worldv2",
    "yolov8l-world",
    "yolov8l-worldv2",
    "yolov8x-world",
    "yolov8x-worldv2",
)


class OwlViTZeroShotModel(Model):
    def __init__(self, config):
        self.checkpoint = "google/owlvit-base-patch32"

        self.candidate_labels = config.get("categories", None)

        from transformers import pipeline

        self.model = pipeline(
            model=self.checkpoint, task="zero-shot-object-detection"
        )

    @property
    def media_type(self):
        return "image"

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def _predict(self, image):
        raw_predictions = self.model(
            image, candidate_labels=self.candidate_labels
        )

        size = image.size
        w, h = size[0], size[1]

        detections = []
        for prediction in raw_predictions:
            score, box = prediction["score"], prediction["box"]
            bounding_box = [
                box["xmin"] / w,
                box["ymin"] / h,
                box["xmax"] / w,
                box["ymax"] / h,
            ]
            ### constrain bounding box to [0, 1]
            bounding_box[0] = max(0, bounding_box[0])
            bounding_box[1] = max(0, bounding_box[1])
            bounding_box[2] = min(1, bounding_box[2])
            bounding_box[3] = min(1, bounding_box[3])

            ### convert to (x, y, w, h)
            bounding_box[2] = bounding_box[2] - bounding_box[0]
            bounding_box[3] = bounding_box[3] - bounding_box[1]

            label = prediction["label"]

            detection = fo.Detection(
                label=label,
                bounding_box=bounding_box,
                confidence=score,
            )
            detections.append(detection)

        return fo.Detections(detections=detections)

    def predict_all(self, samples, args):
        pass


def OwlViT_activator():
    return find_spec("transformers") is not None


def GroundingDINO(config):
    classes = config.get("categories", None)
    model = foz.load_zoo_model(
        "zero-shot-detection-transformer-torch",
        name_or_path="IDEA-Research/grounding-dino-tiny",
        classes=classes,
    )
    return model


def GroundingDINO_activator():
    if find_spec("transformers") is None:
        return False
    required_version = "4.40.0"
    installed_version = pkg_resources.get_distribution("transformers").version
    if installed_version < required_version:
        return False

    required_fiftyone_version = "0.24.0"
    installed_fiftyone_version = pkg_resources.get_distribution(
        "fiftyone"
    ).version
    if installed_fiftyone_version < required_fiftyone_version:
        return False

    return True


def YOLOWorldModel(config):
    classes = config.get("categories", None)
    pretrained = config.get("pretrained", "yolov8l-world")
    if "v2" in pretrained:
        from ultralytics import YOLO

        model = YOLO(pretrained + ".pt")
        model.set_classes(classes)
        import fiftyone.utils.ultralytics as fouu

        model = fouu.convert_ultralytics_model(model)
    else:
        model = foz.load_zoo_model(pretrained + "-torch", classes=classes)
    return model


def YOLOWorld_activator():
    if find_spec("ultralytics") is None:
        return False
    required_version = "8.1.42"
    installed_version = pkg_resources.get_distribution("ultralytics").version
    return installed_version >= required_version


def build_detection_models_dict():
    dms = {}

    if OwlViT_activator():
        dms["OwlViT"] = {
            "activator": OwlViT_activator,
            "model": OwlViTZeroShotModel,
            "submodels": None,
            "name": "OwlViT",
        }

    if YOLOWorld_activator():
        dms["YOLO-World"] = {
            "activator": YOLOWorld_activator,
            "model": YOLOWorldModel,
            "submodels": YOLO_WORLD_PRETRAINS,
            "name": "YOLO-World",
        }

    if GroundingDINO_activator():
        dms["GroundingDINO"] = {
            "activator": GroundingDINO_activator,
            "model": GroundingDINO,
            "submodels": None,
            "name": "GroundingDINO",
        }

    return dms


DETECTION_MODELS = build_detection_models_dict()


def _get_model(model_name, config):
    return DETECTION_MODELS[model_name]["model"](config)


def run_zero_shot_detection(
    dataset, model_name, label_field, categories, pretrained=None, **kwargs
):
    confidence = kwargs.get("confidence", 0.2)
    config = {"categories": categories, "pretrained": pretrained}
    model = _get_model(model_name, config)
    dataset.apply_model(
        model, label_field=label_field, confidence_thresh=confidence
    )
