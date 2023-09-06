"""Zero Shot Semantic Segmentation.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
from PIL import Image
import torch

import fiftyone as fo
from fiftyone.core.models import Model


class CLIPSegZeroShotModel(Model):
    def __init__(self, config):
        self.candidate_labels = config.get("categories", None)

        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    @property
    def media_type(self):
        return "image"    

    def _predict(self, image):
        inputs = self.processor(text=self.candidate_labels, images=[image] * len(self.candidate_labels), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        mask = torch.argmax(preds, dim=0).squeeze().numpy()
        return fo.Segmentation(mask=mask)
    
    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions
 
    def predict_all(self, samples, args):
        pass


def CLIPSeg_activator():
    return find_spec("transformers") is not None



SEMANTIC_SEGMENTATION_MODELS = {
    "CLIPSeg": {
        "activator": CLIPSeg_activator,
        "model": CLIPSegZeroShotModel,
        "name": "CLIPSeg",
    }
}


def _get_model(model_name, config):
    return SEMANTIC_SEGMENTATION_MODELS[model_name]["model"](config)


def run_zero_shot_semantic_segmentation(dataset, model_name, label_field, categories):
    if "other" not in categories:
        categories.append("other")
    config = {"categories": categories}
    model = _get_model(model_name, config)
    dataset.apply_model(model, label_field=label_field)

    dataset.mask_targets[label_field] = {
        i: label for i, label in enumerate(categories)
    }
    dataset.save()