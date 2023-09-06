"""Zero Shot Prediction plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import base64
import sys

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    # pylint: disable=no-name-in-module,import-error
    from classification import (
        run_zero_shot_classification,
        CLASSIFICATION_MODELS,
    )
    from detection import run_zero_shot_detection, DETECTION_MODELS
    from instance_segmentation import (
        run_zero_shot_instance_segmentation,
        INSTANCE_SEGMENTATION_MODELS,
    )
    from semantic_segmentation import (
        run_zero_shot_semantic_segmentation,
        SEMANTIC_SEGMENTATION_MODELS,
    )


ZERO_SHOT_TASKS = (
    "classification",
    "detection",
    "instance_segmentation",
    "semantic_segmentation",
)


MODEL_LISTS = {
    "classification": CLASSIFICATION_MODELS,
    "detection": DETECTION_MODELS,
    "instance_segmentation": INSTANCE_SEGMENTATION_MODELS,
    "semantic_segmentation": SEMANTIC_SEGMENTATION_MODELS,
}


def _get_active_models(task):
    ams = []
    for element in MODEL_LISTS[task].values():
        if element["activator"]():
            ams.append(element["name"])
    return ams


def _get_labels(ctx):
    if ctx.params.get("label_input_choices", False) == "direct":
        labels = ctx.params.get("labels", "")
        return [label.strip() for label in labels.split(",")]
    else:
        labels_file = ctx.params.get("labels_file", None).strip()
        decoded_bytes = base64.b64decode(labels_file.split(",")[1])
        labels = decoded_bytes.decode("utf-8")
        return [label.strip() for label in labels.split("\n")]


TASK_TO_FUNCTION = {
    "classification": run_zero_shot_classification,
    "detection": run_zero_shot_detection,
    "instance_segmentation": run_zero_shot_instance_segmentation,
    "semantic_segmentation": run_zero_shot_semantic_segmentation,
}


def run_zero_shot_task(dataset, task, model_name, label_field, categories):
    return TASK_TO_FUNCTION[task](dataset, model_name, label_field, categories)


class ZeroShotTasks(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zero_shot_predict",
            label="Perform Zero Shot Prediction",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        radio_choices = types.RadioGroup()
        radio_choices.add_choice("classification", label="Classification")
        radio_choices.add_choice("detection", label="Detection")
        radio_choices.add_choice(
            "instance_segmentation", label="Instance Segmentation"
        )
        radio_choices.add_choice(
            "semantic_segmentation", label="Semantic Segmentation"
        )
        inputs.enum(
            "task_choices",
            radio_choices.values(),
            default=radio_choices.choices[0].value,
            label="Zero Shot Task",
            view=radio_choices,
        )

        chosen_task = ctx.params.get("task_choices", "classification")
        active_models = _get_active_models(chosen_task)

        if len(active_models) == 0:
            inputs.str(
                "no_models_warning",
                view=types.Warning(
                    label=f"No Models Found",
                    description="No models were found for the selected task. Please install the required libraries.",
                ),
            )
            return types.Property(inputs)

        model_dropdown = types.Dropdown(
            label=f"{chosen_task.capitalize()} Model"
        )
        for model in active_models:
            model_dropdown.add_choice(model, label=model)
        inputs.enum(
            "model_choice",
            model_dropdown.values(),
            default=model_dropdown.choices[0].value,
            view=model_dropdown,
        )

        label_input_choices = types.RadioGroup()
        label_input_choices.add_choice("direct", label="Input directly")
        label_input_choices.add_choice("file", label="Input from file")
        inputs.enum(
            "label_input_choices",
            label_input_choices.values(),
            default=label_input_choices.choices[0].value,
            label="Labels",
            view=label_input_choices,
        )

        if ctx.params.get("label_input_choices", False) == "direct":
            inputs.str(
                "labels",
                label="Labels",
                description="Enter the names of the classes you wish to generate predictions for, separated by commas",
                required=True,
            )
        else:
            labels_file = types.FileView(label="Labels File")
            inputs.str(
                "labels_file",
                label="Labels File",
                required=True,
                view=labels_file,
            )

        model_name = ctx.params.get("model_choice", "CLIP")
        inputs.str(
            "label_field",
            label="Label Field",
            default=model_name,
            description="The field to store the predicted labels in",
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        dataset = ctx.dataset
        task = ctx.params.get("task_choices", "classification")
        model_name = ctx.params.get("model_choice", "CLIP")
        categories = _get_labels(ctx)
        label_field = ctx.params.get("label_field", model_name)
        run_zero_shot_task(dataset, task, model_name, label_field, categories)
        ctx.trigger("reload_dataset")


### Common input control flow for all tasks
def _input_control_flow(ctx, task):
    inputs = types.Object()
    active_models = _get_active_models(task)
    if len(active_models) == 0:
        inputs.str(
            "no_models_warning",
            view=types.Warning(
                label=f"No Models Found",
                description="No models were found for the selected task. Please install the required libraries.",
            ),
        )
        return types.Property(inputs)

    model_dropdown = types.Dropdown(label=f"{task.capitalize()} Model")
    for model in active_models:
        model_dropdown.add_choice(model, label=model)
    inputs.enum(
        "model_choice",
        model_dropdown.values(),
        default=model_dropdown.choices[0].value,
        view=model_dropdown,
    )

    label_input_choices = types.RadioGroup()
    label_input_choices.add_choice("direct", label="Input directly")
    label_input_choices.add_choice("file", label="Input from file")
    inputs.enum(
        "label_input_choices",
        label_input_choices.values(),
        default=label_input_choices.choices[0].value,
        label="Labels",
        view=label_input_choices,
    )

    if ctx.params.get("label_input_choices", False) == "direct":
        inputs.str(
            "labels",
            label="Labels",
            description="Enter the names of the classes you wish to generate predictions for, separated by commas",
            required=True,
        )
    else:
        labels_file = types.FileView(label="Labels File")
        inputs.str(
            "labels_file",
            label="Labels File",
            required=True,
            view=labels_file,
        )

    model_name = ctx.params.get(
        "model_choice", model_dropdown.choices[0].value
    )
    inputs.str(
        "label_field",
        label="Label Field",
        default=model_name.lower().replace(" ", "_"),
        description="The field to store the predicted labels in",
        required=True,
    )
    return inputs


def _execute_control_flow(ctx, task):
    dataset = ctx.dataset
    model_name = ctx.params.get("model_choice", "CLIP")
    categories = _get_labels(ctx)
    label_field = ctx.params.get("label_field", model_name)
    run_zero_shot_task(dataset, task, model_name, label_field, categories)
    ctx.trigger("reload_dataset")


class ZeroShotClassify(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zero_shot_classify",
            label="Perform Zero Shot Classification",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "classification")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "classification")


class ZeroShotDetect(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zero_shot_detect",
            label="Perform Zero Shot Detection",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "detection")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "detection")


class ZeroShotInstanceSegment(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zero_shot_instance_segment",
            label="Perform Zero Shot Instance Segmentation",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "instance_segmentation")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "instance_segmentation")


class ZeroShotSemanticSegment(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zero_shot_semantic_segment",
            label="Perform Zero Shot Semantic Segmentation",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "semantic_segmentation")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "semantic_segmentation")


def register(plugin):
    plugin.register(ZeroShotTasks)
    plugin.register(ZeroShotClassify)
    plugin.register(ZeroShotDetect)
    plugin.register(ZeroShotInstanceSegment)
    plugin.register(ZeroShotSemanticSegment)
