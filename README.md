## Zero Shot Prediction Plugin

This plugin allows you to perform zero-shot prediction on your dataset for the following tasks:

- Image Classification
- Object Detection
- Instance Segmentation
- Semantic Segmentation

Given a list of label classes, which you can input either manually, separated by commas, or by uploading a text file, the plugin will perform zero-shot prediction on your dataset for the specified task and add the results to the dataset under a new field, which you can specify.

## Models

### Built-in Models

As a starting point, this plugin comes with at least one zero-shot model per task. These are:

- Image Classification: [CLIP](https://github.com/openai/CLIP)
- Object Detection: [Owl-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)
- Instance Segmentation: [Owl-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) + [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- Semantic Segmentation: [CLIPSeg](https://huggingface.co/blog/clipseg-zero-shot)

The Owl-ViT and CLIPSeg models used are from the [HuggingFace Transformers](https://huggingface.co/transformers/) library, and the CLIP and SAM models are from the [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html).

### Adding Your Own Models

You can see the implementations for all of these models in the following files:

- `classification.py`
- `detection.py`
- `instance_segmentation.py`
- `semantic_segmentation.py`

These models are "registered" via dictionaries in each file. In `classification.py`, for example, the dictionary is:

```py
CLASSIFICATION_MODELS = {
    "CLIP": {
        "activator": CLIP_activator,
        "model": CLIPZeroShotModel,
        "name": "CLIP",
    }
}
```

The `activator` checks the environment to see if the model is available, and the `model` is a `fiftyone.core.models.Model` object that is instantiated with the model name and the task. The `name` is the name of the model that will be displayed in the dropdown menu in the plugin.

If you want to add your own model, you can add it to the dictionary in the corresponding file. For example, if you want to add a new image classification model, you can add it to the `CLASSIFICATION_MODELS` dictionary in `classification.py`:

```py
CLASSIFICATION_MODELS = {
    "CLIP": {
        "activator": CLIP_activator,
        "model": CLIPZeroShotModel,
        "name": "CLIP",
    },
    "My Model": {
        "activator": my_model_activator,
        "model": my_model,
        "name": "My Model",
    }
}
```

💡 You need to implement the `activator` and `model` functions for your model. The `activator` should check the environment to see if the model is available, and the `model` should be a `fiftyone.core.models.Model` object that is instantiated with the model name and the task.

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/zero-shot-prediction-plugin
```

## Operators

### `zero_shot_predict`

- Select the task you want to perform zero-shot prediction on (image classification, object detection, instance segmentation, or semantic segmentation), and the field you want to add the results to.

### `zero_shot_classify`

- Perform zero-shot image classification on your dataset

### `zero_shot_detect`

- Perform zero-shot object detection on your dataset

### `zero_shot_instance_segment`

- Perform zero-shot instance segmentation on your dataset

### `zero_shot_semantic_segment`

- Perform zero-shot semantic segmentation on your dataset
