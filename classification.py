"""Zero Shot Classification.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from importlib.util import find_spec
import numpy as np
from PIL import Image
import torch

import fiftyone as fo
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


class AltCLIPZeroShotModel(Model):
    def __init__(self, config):
        self.categories = config.get("categories", None)
        self.candidate_labels = [
            f"a photo of a {cat}" for cat in self.categories
        ]

        from transformers import AltCLIPModel, AltCLIPProcessor

        self.model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        self.processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        inputs = self.processor(
            text=self.candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).numpy()

        return fo.Classification(
            label=self.categories[probs.argmax()],
            logits=logits_per_image.squeeze().numpy(),
            confidence=np.amax(probs[0]),
        )

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def AltCLIP_activator():
    return find_spec("transformers") is not None


class MetaCLIPZeroShotModel(Model):
    def __init__(self, config):
        self.categories = config.get("categories", None)
        self.candidate_labels = [
            f"a photo of a {cat}" for cat in self.categories
        ]

        from transformers import pipeline

        self.pipe = pipeline(
            "zero-shot-image-classification",
            model="facebook/metaclip-h14-fullcc2.5b",
        )

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        res = self.pipe([image], candidate_labels=self.candidate_labels)
        probs = np.array([r["score"] for r in res[0]])
        labels = np.array([r["label"] for r in res[0]])
        lc = labels[0][13:]
        return fo.Classification(
            label=lc,
            logits=probs,
            confidence=np.amax(probs),
        )

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def MetaCLIP_activator():
    return find_spec("transformers") is not None


class AlignZeroShotModel(Model):
    def __init__(self, config):
        self.categories = config.get("categories", None)
        self.candidate_labels = [
            f"a photo of a {cat}" for cat in self.categories
        ]

        from transformers import AlignProcessor, AlignModel

        self.processor = AlignProcessor.from_pretrained(
            "kakaobrain/align-base"
        )
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        inputs = self.processor(
            text=self.candidate_labels, images=image, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).numpy()

        return fo.Classification(
            label=self.categories[probs.argmax()],
            logits=logits_per_image.squeeze().numpy(),
            confidence=np.amax(probs[0]),
        )

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def Align_activator():
    return find_spec("transformers") is not None


class OpenCLIPZeroShotModel(Model):
    def __init__(self, config):
        self.categories = config.get("categories", None)
        self.candidate_labels = [
            f"a photo of a {cat}" for cat in self.categories
        ]

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        tokenizer = open_clip.get_tokenizer(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        self.model = model
        self._preprocess = preprocess
        self.tokenizer = tokenizer

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        image = self._preprocess(image).unsqueeze(0)
        text = self.tokenizer(self.candidate_labels)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (
                (100.0 * image_features @ text_features.T)
                .softmax(dim=-1)[0]
                .numpy()
            )

        lc = self.categories[np.argmax(text_probs)]

        return fo.Classification(
            label=lc,
            logits=text_probs,
            confidence=np.amax(text_probs),
        )

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def OpenCLIP_activator():
    return find_spec("open_clip") is not None


CLASSIFICATION_MODELS = {
    "CLIP": {
        "activator": CLIP_activator,
        "model": CLIPZeroShotModel,
        "name": "CLIP",
    },
    "AltCLIP": {
        "activator": AltCLIP_activator,
        "model": AltCLIPZeroShotModel,
        "name": "AltCLIP",
    },
    "MetaCLIP-H14": {
        "activator": MetaCLIP_activator,
        "model": MetaCLIPZeroShotModel,
        "name": "MetaCLIP-H14",
    },
    "Align": {
        "activator": Align_activator,
        "model": AlignZeroShotModel,
        "name": "Align",
    },
    "OpenCLIP-ViT-H-14-laion2B": {
        "activator": OpenCLIP_activator,
        "model": OpenCLIPZeroShotModel,
        "name": "OpenCLIP-ViT-H-14-laion2B",
    },
}


def _get_model(model_name, config):
    return CLASSIFICATION_MODELS[model_name]["model"](config)


def run_zero_shot_classification(dataset, model_name, label_field, categories):
    config = {"categories": categories}
    model = _get_model(model_name, config)
    dataset.apply_model(model, label_field=label_field)
