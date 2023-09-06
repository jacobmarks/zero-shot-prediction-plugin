"""Zero Shot Classification.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from fiftyone.core.models import Model
import fiftyone.zoo as foz


class CLIPZeroShotModel(Model):
    def __init__(self, config):
        cats = config.get("categories", None)
        self.model = foz.load_zoo_model(
            "clip-vit-base32-torch",
            text_prompt="A photo of a",
            classes=cats,
        )

    @property
    def media_type(self):
        return "image"

    def predict(self, args):
        return self.model.predict(args)

    def predict_all(self, samples, args):
        return self.model.predict_all(samples, args)


def CLIP_activator():
    return True


CLASSIFICATION_MODELS = {
    "CLIP": {
        "activator": CLIP_activator,
        "model": CLIPZeroShotModel,
        "name": "CLIP",
    }
}


def _get_model(model_name, config):
    return CLASSIFICATION_MODELS[model_name]["model"](config)


def run_zero_shot_classification(dataset, model_name, label_field, categories):
    config = {"categories": categories}
    model = _get_model(model_name, config)
    dataset.apply_model(model, label_field=label_field)
