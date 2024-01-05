"""Zero Shot Prediction plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import base64

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


def _format_submodel_name(submodel):
    return f"{submodel[0]}|{submodel[1]}"


def _format_submodel_label(submodel):
    pretrain = submodel[0].split("/")[-1]
    arch = submodel[1]

    arch_string = f"Architecture: {arch}"
    _pt = pretrain and pretrain != "openai"
    pretrain_string = f" | Pretrained: {pretrain}" if _pt else ""
    return f"{arch_string}{pretrain_string}"


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def _list_target_views(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = "DATASET"
    if has_view or has_selected:
        target_choices = types.RadioGroup()
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Run model on the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Run model on the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Run model on the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
        )
    else:
        ctx.params["target"] = "DATASET"


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


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
        if "," in labels_file:
            lf = labels_file.split(",")[1]
        else:
            lf = labels_file
        decoded_bytes = base64.b64decode(lf)
        labels = decoded_bytes.decode("utf-8")
        return [label.strip() for label in labels.split("\n")]


TASK_TO_FUNCTION = {
    "classification": run_zero_shot_classification,
    "detection": run_zero_shot_detection,
    "instance_segmentation": run_zero_shot_instance_segmentation,
    "semantic_segmentation": run_zero_shot_semantic_segmentation,
}


def run_zero_shot_task(
    dataset,
    task,
    model_name,
    label_field,
    categories,
    architecture,
    pretrained,
):
    return TASK_TO_FUNCTION[task](
        dataset,
        model_name,
        label_field,
        categories,
        architecture=architecture,
        pretrained=pretrained,
    )


def _model_name_to_field_name(model_name):
    fn = (
        model_name.lower()
        .replace(" ", "_")
        .replace("_+", "")
        .replace("-", "")
        .split("(")[0]
        .strip()
    )
    if fn[-1] == "_":
        fn = fn[:-1]
    return fn


class ZeroShotTasks(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="zero_shot_predict",
            label="Perform Zero Shot Prediction",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

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

        model_dropdown_label = f"{chosen_task.capitalize()} Model"
        model_dropdown = types.Dropdown(label=model_dropdown_label)
        for model in active_models:
            model_dropdown.add_choice(model, label=model)
        inputs.enum(
            f"model_choice_{chosen_task}",
            model_dropdown.values(),
            default=model_dropdown.choices[0].value,
            view=model_dropdown,
        )

        model_choice = ctx.params.get(
            f"model_choice_{chosen_task}", model_dropdown.choices[0].value
        )
        mc = model_choice.split("(")[0].strip().lower()

        submodels = MODEL_LISTS[chosen_task][model_choice].get(
            "submodels", None
        )
        if submodels is not None:
            if len(submodels) == 1:
                ctx.params["pretrained"] = submodels[0][0]
                ctx.params["architecture"] = submodels[0][1]
            else:
                submodel_dropdown = types.Dropdown(
                    label=f"{chosen_task.capitalize()} Submodel"
                )
                for submodel in submodels:
                    submodel_dropdown.add_choice(
                        _format_submodel_name(submodel),
                        label=_format_submodel_label(submodel),
                    )
                inputs.enum(
                    f"submodel_choice_{chosen_task}_{mc}",
                    submodel_dropdown.values(),
                    default=submodel_dropdown.choices[0].value,
                    view=submodel_dropdown,
                )

                submodel_choice = ctx.params.get(
                    f"submodel_choice_{chosen_task}_{model_choice}",
                    submodel_dropdown.choices[0].value,
                )
                ctx.params["pretrained"] = submodel_choice.split("|")[0]
                ctx.params["architecture"] = submodel_choice.split("|")[1]

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

        if ctx.params.get("label_input_choices", "direct") == "direct":
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
            f"model_choice_{chosen_task}", active_models[0]
        )
        model_name = model_name.split("(")[0].strip()
        inputs.str(
            f"label_field_{chosen_task}_{model_name}",
            label="Label Field",
            default=_model_name_to_field_name(model_name),
            description="The field to store the predicted labels in",
            required=True,
        )
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        task = ctx.params.get("task_choices", "classification")
        active_models = _get_active_models(task)
        model_name = ctx.params.get(f"model_choice_{task}", active_models[0])
        mn = model_name.split("(")[0].strip()
        categories = _get_labels(ctx)
        label_field = ctx.params.get(f"label_field_{task}_{mn}", mn)
        architecture = ctx.params.get("architecture", None)
        pretrained = ctx.params.get("pretrained", None)

        with open("/tmp/params.txt", "w") as f:
            f.write(str(ctx.params))
        run_zero_shot_task(
            view,
            task,
            model_name,
            label_field,
            categories,
            architecture,
            pretrained,
        )
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

    model_choice = ctx.params.get(
        f"model_choice", model_dropdown.choices[0].value
    )
    mc = model_choice.split("(")[0].strip().lower()

    submodels = MODEL_LISTS[task][model_choice].get("submodels", None)
    if submodels is not None:
        if len(submodels) == 1:
            ctx.params["pretrained"] = submodels[0][0]
            ctx.params["architecture"] = submodels[0][1]
        else:
            submodel_dropdown = types.Dropdown(
                label=f"{task.capitalize()} Submodel"
            )
            for submodel in submodels:
                submodel_dropdown.add_choice(
                    _format_submodel_name(submodel),
                    label=_format_submodel_label(submodel),
                )
            inputs.enum(
                f"submodel_choice_{task}_{mc}",
                submodel_dropdown.values(),
                default=submodel_dropdown.choices[0].value,
                view=submodel_dropdown,
            )

            submodel_choice = ctx.params.get(
                f"submodel_choice_{task}_{model_choice}",
                submodel_dropdown.choices[0].value,
            )
            ctx.params["pretrained"] = submodel_choice.split("|")[0]
            ctx.params["architecture"] = submodel_choice.split("|")[1]

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
    mn = model_name.split("(")[0].strip().lower()
    inputs.str(
        f"label_field_{mn}",
        label="Label Field",
        default=_model_name_to_field_name(model_name),
        description="The field to store the predicted labels in",
        required=True,
    )
    _execution_mode(ctx, inputs)
    _list_target_views(ctx, inputs)
    return inputs


def _execute_control_flow(ctx, task):
    view = _get_target_view(ctx, ctx.params["target"])
    model_name = ctx.params.get("model_choice", "CLIP")
    categories = _get_labels(ctx)
    label_field = ctx.params.get(
        f"label_field_{model_name}", _model_name_to_field_name(model_name)
    )
    architecture = ctx.params.get("architecture", None)
    pretrained = ctx.params.get("pretrained", None)
    run_zero_shot_task(
        view,
        task,
        model_name,
        label_field,
        categories,
        architecture,
        pretrained,
    )
    ctx.trigger("reload_dataset")


NAME_TO_TASK = {
    "zero_shot_classify": "classification",
    "zero_shot_detect": "detection",
    "zero_shot_instance_segment": "instance_segmentation",
    "zero_shot_semantic_segment": "semantic_segmentation",
}


def _format_model_name(model_name):
    return (
        model_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    )


def _match_model_name(model_name, model_names):
    for name in model_names:
        if _format_model_name(name) == _format_model_name(model_name):
            return name
    raise ValueError(
        f"Model name {model_name} not found. Use one of {model_names}"
    )


def _handle_calling(
    uri,
    sample_collection,
    model_name,
    labels,
    labels_file,
    label_field,
    delegate,
):
    ctx = dict(view=sample_collection.view())

    task = NAME_TO_TASK[uri.split("/")[-1]]

    if model_name is None:
        model_name = list(MODEL_LISTS[task].keys())[0]
    elif model_name not in MODEL_LISTS[task]:
        model_name = _match_model_name(
            model_name, list(MODEL_LISTS[task].keys())
        )

    if labels is None and labels_file is None:
        raise ValueError("Must provide either labels or labels_file")

    if labels is not None and labels_file is not None:
        raise ValueError("Cannot provide both labels and labels_file")

    if labels is not None and type(labels) == list:
        labels = ", ".join(labels)
    else:
        with open(labels_file, "r") as f:
            labels = [label.strip() for label in f.readlines()]
        labels = ", ".join(labels)

    if label_field is None:
        label_field = _model_name_to_field_name(model_name)

    label_field_name = f"label_field_{model_name}"

    params = dict(
        target="CURRENT_VIEW",
        model_choice=model_name,
        label_input_choices="direct",
        delegate=delegate,
        labels=labels,
    )
    params[label_field_name] = label_field

    return foo.execute_operator(uri, ctx, params=params)


class ZeroShotClassify(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="zero_shot_classify",
            label="Perform Zero Shot Classification",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "classification")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "classification")

    def __call__(
        self,
        sample_collection,
        model_name=None,
        labels=None,
        labels_file=None,
        label_field=None,
        delegate=False,
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            labels,
            labels_file,
            label_field,
            delegate,
        )

    def list_models(self):
        return list(MODEL_LISTS["classification"].keys())


class ZeroShotDetect(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="zero_shot_detect",
            label="Perform Zero Shot Detection",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "detection")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "detection")

    def __call__(
        self,
        sample_collection,
        model_name=None,
        labels=None,
        labels_file=None,
        label_field=None,
        delegate=False,
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            labels,
            labels_file,
            label_field,
            delegate,
        )

    def list_models(self):
        return list(MODEL_LISTS["detection"].keys())


class ZeroShotInstanceSegment(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="zero_shot_instance_segment",
            label="Perform Zero Shot Instance Segmentation",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "instance_segmentation")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "instance_segmentation")

    def __call__(
        self,
        sample_collection,
        model_name=None,
        labels=None,
        labels_file=None,
        label_field=None,
        delegate=False,
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            labels,
            labels_file,
            label_field,
            delegate,
        )

    def list_models(self):
        return list(MODEL_LISTS["instance_segmentation"].keys())


class ZeroShotSemanticSegment(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="zero_shot_semantic_segment",
            label="Perform Zero Shot Semantic Segmentation",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = _input_control_flow(ctx, "semantic_segmentation")
        return types.Property(inputs)

    def execute(self, ctx):
        _execute_control_flow(ctx, "semantic_segmentation")

    def __call__(
        self,
        sample_collection,
        model_name=None,
        labels=None,
        labels_file=None,
        label_field=None,
        delegate=False,
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            labels,
            labels_file,
            label_field,
            delegate,
        )

    def list_models(self):
        return list(MODEL_LISTS["semantic_segmentation"].keys())


def register(plugin):
    plugin.register(ZeroShotTasks)
    plugin.register(ZeroShotClassify)
    plugin.register(ZeroShotDetect)
    plugin.register(ZeroShotInstanceSegment)
    plugin.register(ZeroShotSemanticSegment)
