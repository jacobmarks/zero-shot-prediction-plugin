## Zero Shot Prediction Plugin

![zero_shot_owlvit_example](https://github.com/jacobmarks/zero-shot-prediction-plugin/assets/12500356/6aca099a-17b3-4f85-955d-26c3951f0646)

This plugin allows you to perform zero-shot prediction on your dataset for the following tasks:

- Image Classification
- Object Detection
- Instance Segmentation
- Semantic Segmentation

Given a list of label classes, which you can input either manually, separated by commas, or by uploading a text file, the plugin will perform zero-shot prediction on your dataset for the specified task and add the results to the dataset under a new field, which you can specify.

### Updates

- **2023-10-27**: Added support for MetaCLIP for image classification
- **2023-10-20**: Added support for AltCLIP and Align for image classification and GroupViT for semantic segmentation

## Models

### Built-in Models

As a starting point, this plugin comes with at least one zero-shot model per task. These are:

- Image Classification: [CLIP](https://github.com/openai/CLIP), [AltCLIP](https://huggingface.co/docs/transformers/model_doc/altclip), [MetaCLIP](https://huggingface.co/facebook/metaclip-h14-fullcc2.5b), and [Align](https://huggingface.co/docs/transformers/model_doc/align)
- Object Detection: [Owl-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)
- Instance Segmentation: [Owl-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) + [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- Semantic Segmentation: [CLIPSeg](https://huggingface.co/blog/clipseg-zero-shot) and [GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit)

Most of the models used are from the [HuggingFace Transformers](https://huggingface.co/transformers/) library, and CLIP and SAM models are from the [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html)

_Note_— For SAM you will need to have Facebook's `segment-anything` library installed.

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
    ..., # other models
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

If you want to use AltCLIP, Align, Owl-ViT, CLIPSeg, or GroupViT, you will also need to install the `transformers` library:

```shell
pip install transformers
```

If you want to use SAM, you will also need to install the `segment-anything` library:

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Usage

All of the operators in this plugin can be run in _delegated_ execution mode. This means that instead of waiting for the operator to finish, you _schedule_
the operation to be performed separately. This is useful for long-running operations, such as performing inference on a large dataset.

Once you have pressed the `Schedule` button for the operator, you will be able to see the job from the command line using FiftyOne's [command line interface](https://docs.voxel51.com/cli/index.html#fiftyone-delegated-operations):

```shell
fiftyone delegated list
```

will show you the status of all delegated operations.

To launch a service which runs the operation, as well as any other delegated operations that have been scheduled, run:

```shell
fiftyone delegated launch
```

Once the operation has completed, you can view the results in the App (upon refresh).

After the operation completes, you can also clean up your list of delegated operations by running:

```shell
fiftyone delegated cleanup -s COMPLETED
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
