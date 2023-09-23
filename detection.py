"""Zero Shot Detection.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
from PIL import Image

import fiftyone as fo
from fiftyone.core.models import Model


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


DETECTION_MODELS = {
    "OwlViT": {
        "activator": OwlViT_activator,
        "model": OwlViTZeroShotModel,
        "name": "OwlViT",
    }
}


def _get_model(model_name, config):
    return DETECTION_MODELS[model_name]["model"](config)


def run_zero_shot_detection(dataset, model_name, label_field, categories):
    config = {"categories": categories}
    model = _get_model(model_name, config)
    dataset.apply_model(model, label_field=label_field)
