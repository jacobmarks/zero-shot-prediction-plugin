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

AIMV2_MODELS = [
    ("apple-aimv2", "aimv2-large-patch14-native"),
    ("apple-aimv2", "aimv2-large-patch14-224-lit"),
]

def CLIPZeroShotModel(config):
    """
    This function loads a zero-shot classification model using the CLIP architecture.
    It utilizes the FiftyOne Zoo to load a pre-trained model based on the provided
    configuration.

    Args:
        config (dict): A dictionary containing configuration parameters for the model.
            - categories (list, optional): A list of categories for classification.
            - clip_model (str, optional): The architecture of the CLIP model to use.
            - pretrained (str, optional): The pre-trained weights to use.

    Returns:
        Model: A loaded CLIP zero-shot classification model ready for inference.
    """
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
    """
    Determines if the CLIP model can be activated.

    This function checks for the availability of the necessary
    components to activate the CLIP model. It returns True if
    the model can be activated, otherwise False.

    Returns:
        bool: True if the CLIP model can be activated, False otherwise.
    """
    return True

class AltCLIPZeroShotModel(Model):
    """
    This class implements a zero-shot classification model using the AltCLIP architecture.
    It leverages the AltCLIP model from the Hugging Face Transformers library to perform
    image classification without requiring task-specific training data.

    Args:
        config (dict): A dictionary containing configuration parameters for the model.
            - categories (list, optional): A list of categories for classification.

    Attributes:
        categories (list): The list of categories for classification.
        candidate_labels (list): A list of text prompts for each category.
        model (AltCLIPModel): The pre-trained AltCLIP model.
        processor (AltCLIPProcessor): The processor for preparing inputs for the model.

    Methods:
        media_type: Returns the type of media the model is designed to process.
        _predict(image): Performs prediction on a single image.
        predict(args): Converts input data to an image and performs prediction.
        _predict_all(images): Performs prediction on a list of images.
    """
    def __init__(self, config):
        """
        Initializes the AltCLIPZeroShotModel with the given configuration.

        This constructor sets up the model by initializing the categories
        and candidate labels for classification. It also loads the pre-trained
        AltCLIP model and processor from the Hugging Face Transformers library.

        Args:
            config (dict): A dictionary containing configuration parameters for the model.
                - categories (list, optional): A list of categories for classification.
        """
        self.categories = config.get("categories", None)
        self.candidate_labels = [
            f"a photo of a {cat}" for cat in self.categories
        ]

        from transformers import AltCLIPModel, AltCLIPProcessor

        self.model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        self.processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move model to appropriate device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        """
        Performs prediction on a single image.

        This method processes the input image using the AltCLIPProcessor
        and performs a forward pass through the AltCLIPModel to obtain
        classification probabilities for each category. It returns a
        FiftyOne Classification object containing the predicted label,
        logits, and confidence score.

        Args:
            image (PIL.Image): The input image to classify.

        Returns:
            fo.Classification: The classification result containing the
            predicted label, logits, and confidence score.
        """
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
        """
        Converts input data to an image and performs prediction.

        This method takes input data, converts it into a PIL image,
        and then uses the `_predict` method to perform classification.
        It returns the prediction results as a FiftyOne Classification object.

        Args:
            args (numpy.ndarray): The input data to be converted into an image.

        Returns:
            fo.Classification: The classification result containing the
            predicted label, logits, and confidence score.
        """
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def _predict_all(self, images):
        return [self._predict(image) for image in images]


def AltCLIP_activator():
    return find_spec("transformers") is not None


class AlignZeroShotModel(Model):
    """
    AlignZeroShotModel is a class for zero-shot image classification using the Align model.

    This class leverages the Align model from the `transformers` library to perform
    zero-shot classification on images. It initializes with a configuration that
    specifies the categories for classification. The model processes input images
    and predicts the most likely category from the provided list.

    Attributes:
        categories (list): A list of category labels for classification.
        candidate_labels (list): A list of formatted labels for the Align model.
        processor (AlignProcessor): The processor for preparing inputs for the model.
        model (AlignModel): The pre-trained Align model for classification.

    Methods:
        media_type: Returns the type of media the model works with, which is "image".
        _predict(image): Performs prediction on a single image.
        predict(args): Converts input data to an image and performs prediction.
        _predict_all(images): Performs prediction on a list of images.
    """
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

        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move model to appropriate device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        """
        Performs prediction on a single image.

        This method takes an image as input and processes it using the Align model
        to predict the most likely category from the pre-defined list of categories.
        It uses the processor to prepare the input and the model to generate predictions.
        The method returns a `fiftyone.core.labels.Classification` object containing
        the predicted label, logits, and confidence score.

        Args:
            image (PIL.Image.Image): The input image for classification.

        Returns:
            fiftyone.core.labels.Classification: The classification result with the
            predicted label, logits, and confidence score.
        """
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
        """
        Predicts the category of the given image.

        This method takes an image in the form of a numpy array, converts it
        to a PIL Image, and then uses the `_predict` method to classify the
        image. The classification result is returned as a 
        `fiftyone.core.labels.Classification` object.

        Args:
            args (np.ndarray): The input image as a numpy array.

        Returns:
            fiftyone.core.labels.Classification: The classification result with
            the predicted label, logits, and confidence score.
        """
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def _predict_all(self, images):
        return [self._predict(image) for image in images]


def Align_activator():
    return find_spec("transformers") is not None


def OpenCLIPZeroShotModel(config):
    """
    Initializes and returns an OpenCLIP zero-shot model based on the provided configuration.

    This function loads a pre-trained OpenCLIP model using the specified configuration
    parameters. The model is initialized with a text prompt and a set of categories
    for zero-shot classification tasks.

    Args:
        config (dict): A dictionary containing configuration parameters for the model.
            - "categories" (list, optional): A list of category names for classification.
            - "clip_model" (str, optional): The name of the CLIP model architecture to use.
              Defaults to "ViT-B-32".
            - "pretrained" (str, optional): The name of the pre-trained weights to load.
              Defaults to "openai".

    Returns:
        An instance of the OpenCLIP model configured for zero-shot classification.
    """
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
    """
    Builds a dictionary of classification models available for use.

    This function constructs a dictionary where each key is a string representing
    the name of a classification model type, and the value is a dictionary containing
    the following keys:
        - "activator": A function that checks if the model's dependencies are available.
        - "model": A function that initializes and returns the model.
        - "submodels": Additional model configurations or variants, if any.
        - "name": The display name of the model.

    The function first checks if the OpenCLIP library is available. If not, it adds
    the "CLIP (OpenAI)" model to the dictionary and returns it. If OpenCLIP is available,
    it proceeds to add other models like "ALIGN" and "AltCLIP" based on their respective
    activators. Finally, it iterates over the `CLASSIFICATION_MODEL_TYPES` to add
    models that use the OpenCLIP framework.

    Returns:
        dict: A dictionary of available classification models.
    """
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
