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

### Make tuples in form ("pretrained", "clip_model")

OPENAI_ARCHS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENAI_CLIP_MODELS = [("openai", model) for model in OPENAI_ARCHS]

DFN_CLIP_MODELS = [("dfn2b", "ViT-B-16")]

META_ARCHS = ("ViT-B-16-quickgelu", "ViT-B-32-quickgelu", "ViT-L-14-quickgelu")
META_PRETRAINS = ("metaclip_400m", "metaclip_fullcc")
META_CLIP_MODELS = [
    (pretrain, arch) for pretrain in META_PRETRAINS for arch in META_ARCHS
]
META_CLIP_MODELS.append(("metaclip_fullcc", "ViT-H-14-quickgelu"))

CLIPA_MODELS = [("", "hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B")]

SIGLIP_ARCHS = (
    "ViT-B-16-SigLIP",
    "ViT-B-16-SigLIP-256",
    "ViT-B-16-SigLIP-384",
    "ViT-L-16-SigLIP-256",
    "ViT-L-16-SigLIP-384",
    "ViT-SO400M-14-SigLIP",
    "ViT-SO400M-14-SigLIP-384",
)
SIGLIP_MODELS = [("", arch) for arch in SIGLIP_ARCHS]

EVA_CLIP_MODELS = [
    ("merged2b_s8b_b131k", "EVA02-B-16"),
    ("merged2b_s6b_b61k", "EVA02-L-14-336"),
    ("merged2b_s4b_b131k", "EVA02-L-14"),
]


def CLIPZeroShotModel(config):
    cats = config.get("categories", None)
    clip_model = config.get("clip_model", "ViT-B-32")
    pretrained = config.get("pretrained", "openai")

    model = foz.load_zoo_model(
        "clip-vit-base32-torch",
        clip_model=clip_model,
        pretrained=pretrained,
        text_prompt="A photo of a",
        classes=cats,
    )

    return model


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

    def _predict_all(self, images):
        return [self._predict(image) for image in images]


def AltCLIP_activator():
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

    def _predict_all(self, images):
        return [self._predict(image) for image in images]


def Align_activator():
    return find_spec("transformers") is not None


def OpenCLIPZeroShotModel(config):
    cats = config.get("categories", None)
    clip_model = config.get("clip_model", "ViT-B-32")
    pretrained = config.get("pretrained", "openai")

    model = foz.load_zoo_model(
        "open-clip-torch",
        clip_model=clip_model,
        pretrained=pretrained,
        text_prompt="A photo of a",
        classes=cats,
    )

    return model


def OpenCLIP_activator():
    return find_spec("open_clip") is not None


CLASSIFICATION_MODEL_TYPES = {
    "CLIP (OpenAI)": OPENAI_CLIP_MODELS,
    "CLIPA": CLIPA_MODELS,
    "DFN CLIP": DFN_CLIP_MODELS,
    "EVA-CLIP": EVA_CLIP_MODELS,
    "MetaCLIP": META_CLIP_MODELS,
    "SigLIP": SIGLIP_MODELS,
}


def build_classification_models_dict():
    cms = {}

    if not OpenCLIP_activator():
        cms["CLIP (OpenAI)"] = {
            "activator": CLIP_activator,
            "model": CLIPZeroShotModel,
            "submodels": None,
            "name": "CLIP (OpenAI)",
        }
        return cms

    if Align_activator():
        cms["ALIGN"] = {
            "activator": Align_activator,
            "model": AlignZeroShotModel,
            "submodels": None,
            "name": "ALIGN",
        }

    if AltCLIP_activator():
        cms["AltCLIP"] = {
            "activator": AltCLIP_activator,
            "model": AltCLIPZeroShotModel,
            "submodels": None,
            "name": "AltCLIP",
        }

    for key, value in CLASSIFICATION_MODEL_TYPES.items():
        cms[key] = {
            "activator": OpenCLIP_activator,
            "model": OpenCLIPZeroShotModel,
            "submodels": value,
            "name": key,
        }

    return cms


CLASSIFICATION_MODELS = build_classification_models_dict()


def _get_model(model_name, config):
    return CLASSIFICATION_MODELS[model_name]["model"](config)


def run_zero_shot_classification(
    dataset,
    model_name,
    label_field,
    categories,
    architecture=None,
    pretrained=None,
    **kwargs,
):
    config = {
        "categories": categories,
        "clip_model": architecture,
        "pretrained": pretrained,
    }

    model = _get_model(model_name, config)

    dataset.apply_model(model, label_field=label_field)
