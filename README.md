# Leap Interpretability Engine

Congratulations on being a _very_ early adopter of our interpretability engine! Not sure what's going on? Check out the [FAQ](#faq).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install leap-ie.

```bash
pip install leap-ie
```

Sign in and generate your API key in the [leap app](https://app.leap-labs.com/) - you'll need this to get started.

## Usage
Using the interpretability engine is really easy! All you need to do is import leap_ie, and wrap your model in our generate function:
```python
results = engine.generate(
    project_name="interpretability",
    model=your_model,
    class_list=["hotdog", "not_hotdog"],
    config={"leap_api_key": "YOUR_LEAP_API_KEY", "input_dim": [3, 224, 224]},
)
```
Currently we support image classification models only. We expect the model to take a batch of images as input, and return a batch of logits (NOT probabilities). For best results, you might have to tune the [config](#config) a bit.

## Results

The generate function returns a pandas dataframe, containing [prototypes](#what-is-a-prototype), [entanglements](#what-is-entanglement), and [feature isolations](#what-is-feature-isolation). If used with samples (see [Sample Feature Isolation](#sample-feature-isolation)), the dataframe contains feature isolations for each sample, for the target classes (if provided), or for the top 3 predicted classes.

If you're in a jupyter notebook, you can view these inline using `engine.display_results(results)`, but for the best experience we recommend you head to the [leap app](https://app.leap-labs.com/) to view your prototypes and isolations, or [log directly to your weights and biases dashboard](#weights-and-biases-integration).

## Supported Frameworks

We support both pytorch and tensorflow! Specify your package with the `mode` parameter, using `'tf'` for tensorflow and `'pt'` for pytorch. (Defaults to pytorch if unspecified.) Tensorflow is still faily experimental and will likely require a fair amount of config tuning - sorry! We're working on it.

If using pytorch, we expect the model to take images to be in channels first format, e.g. of shape `[1, channels, height, width]`. If tensorflow, channels last, e.g.`[1, height, width, channels]`.

## Weights and Biases Integration
We can also log results directly to your WandB projects! To do this, set `project_name` to the name of the WandB project where you'd like the results to be logged, and add your WandB API key and entity name to the `config` dictionary:
```python
config = {
    "wandb_api_key": "YOUR_WANDB_API_KEY",
    "wandb_entity": "your_wandb_entity",
    "leap_api_key": "YOUR_LEAP_API_KEY",
    "input_dim": [3, 224, 224],
}
results = engine.generate(
    project_name="your_wandb_project_name",
    model=your_model,
    class_list=["hotdog", "not_hotdog"],
    config=config,
)
```

## Prototype Generation

Given your model, we generate [prototypes](#what-is-a-prototype) and [entanglements](#what-is-entanglement) We also [isolate entangled features](#what-is-feature-isolation) in your prototypes.

```python
from leap_ie import engine
from leap_ie.models import get_model

config = {"leap_api_key": "YOUR_LEAP_API_KEY"}

# Replace this model with your own, or explore any imagenet classifier from torchvision (https://pytorch.org/vision/stable/models.html).
model = preprocessing_fn, model, class_list = get_model("torchvision.resnet18")

# indexes of classes to generate prototypes for. In this case, ['tench', 'goldfish', 'great white shark'].
target_classes = [0, 1, 2]

# generate prototypes
prototypes = engine.generate(
    project_name="resnet18",
    model=model,
    class_list=class_list,
    config=config,
    target_classes=target_classes,
    preprocessing=preprocessing_fn,
    samples=None,
    device=None,
    mode="pt",
)

# For the best experience, head to https://app.leap-labs.com/ to explore your prototypes and feature isolations in the browser!
# Or, if you're in a jupyter notebook, you can display your results inline:
engine.display_results(prototypes)
```

## Multiple Prototype Generation

To generate multiple prototypes for the same target class, simply repeat the index of the target class, e.g.

```python
target_classes = [0, 0, 0]
```
will generate three prototypes for the 0th class.

## Sample Feature Isolation

Given some input image, we can show you which features your model thinks belong to each class. If you specify target classes, we'll isolate features for those, or if not, we'll isolate features for the three highest probability classes.

```python
from torchvision import transforms
from leap_ie import engine
from leap_ie.models import get_model
from PIL import Image

config = {"leap_api_key": "YOUR_LEAP_API_KEY"}

# Replace this model with your own, or explore any imagenet classifier from torchvision (https://pytorch.org/vision/stable/models.html).
model = preprocessing_fn, model, class_list = get_model("torchvision.resnet18")

# load an image
image_path = "tools.jpeg"
tt = transforms.ToTensor()
image = preprocessing_fn[0](tt(Image.open(image_path)).unsqueeze(0))

# to isolate features:
isolations = engine.generate(
    project_name="resnet18",
    model=model,
    class_list=class_list,
    config=config,
    target_classes=None,
    preprocessing=preprocessing_fn,
    samples=image,
    mode="pt",
)

# For the best experience, head to https://app.leap-labs.com/ to explore your prototypes and feature isolations in the browser!
# Or, if you're in a jupyter notebook, you can display your results inline:
engine.display_results(isolations)
```

## engine.generate()

The generate function is used for both prototype generation directly from the model, and for feature isolation on your input samples.


```python
leap_ie.engine.generate(
    project_name,
    model,
    class_list,
    config,
    target_classes=None,
    preprocessing=None,
    samples=None,
    device=None,
    mode="pt",
)
```

- **project_name** (`str`): Name of your project. Used for logging.
  - *Required*: Yes
  - *Default*: None

- **model** (`object`): Model for interpretation. Currently we support image classification models only. We expect the model to take a batch of images as input, and return a batch of logits (NOT probabilities). If using pytorch, we expect the model to take images to be in channels first format, e.g. of shape `[1, channels, height, width]`. If tensorflow, channels last, e.g.`[1, height, width, channels]`.
  - *Required*: Yes
  - *Default*: None

- **class_list** (`list`): List of class names corresponding to your model's output classes, e.g. ['hotdog', 'not hotdog', ...].
  - *Required*: Yes
  - *Default*: None

- **config** (`dict` or `str`): Configuration dictionary, or path to a json file containing your configuration. At minimum, this must contain `{"leap_api_key": "YOUR_LEAP_API_KEY"}`.
  - *Required*: Yes
  - *Default*: None

- **target_classes** (`list`, optional): List of target class indices to generate prototypes or isolations for, e.g. `[0,1]`. If None, prototypes will be generated for the class at output index 0 only, e.g. 'hotdog', and feature isolations will be generated for the top 3 classes.
  - *Required*: No
  - *Default*: None

- **preprocessing** (`function`, optional): Preprocessing function to be used for generation. This can be None, but for best results, use the preprocessing function used on inputs for inference.
  - *Required*: No
  - *Default*: None

- **samples** (`array`, optional): None, or a batch of images to perform feature isolation on. If provided, only feature isolation is performed (not prototype generation). We expect samples to be of shape `[num_images, height, width, channels]` if using tensorflow, or `[1, channels, height, width]` if using pytorch.
  - *Required*: No
  - *Default*: None

- **device** (`str`, optional): Device to be used for generation. If None, we will try to find a device.
  - *Required*: No
  - *Default*: None

- **mode** (`str`, optional): Framework to use, either 'pt' for pytorch or 'tf' for tensorflow. Default is 'pt'.
  - *Required*: No
  - *Default*: `pt`


## Config

Leap provides a number of configuration options to fine-tune the interpretability engine's performance with your models. You can provide it as a dictionary or a path to a .json file.

Typically, you'll only change a few of these – though feel free to experiment! The key ones are as follows:

- **hf_weight** (`int`): How much to penalise high-frequency patterns in the input. If you are generating very blurry and indistinct prototypes, decrease this. If you are getting very noisy prototypes, increase it. This depends on your model architecture and is hard for us to predict, so you might want to experiment. It's a bit like focussing binoculars. Best practice is to start with zero, and gradually increase.
  - *Default*: `1`

- **input_dim** (`list`): The dimensions of the input that your model expects.
  - *Default*: `[224, 224, 3]` if mode is "tf" else `[3, 224, 224]`

- **isolation** (`bool`): Whether to isolate features for entangled classes. Set to False if you want prototypes only.
  - *Default*: `True`

- **lr** (`float`): How much to update the prototype at each step during the prototype generation process. This can be tuned, but in practice is to around 1% of the expected input range. E.g. if your model was trained on images in the range -1 to 1 (prior to any preprocessing function), 0.02 is a good place to start.
  - *Default*: `0.005`

- **max_steps** (`int`): How many steps to run the prototype generation/feature isolation process for. If you get indistinct prototypes or isolations, try increasing this number.
  - *Default*: `1000`


Here are all of the config options currently available:

```python
config = {
    "use_alpha": False,
    "alpha_mask": False,
    "alpha_only": False,
    "baseline_init": 0,
    "diversity_weight": 0,
    "isolate_classes": None,
    "isolation_lr": 0.05,
    "hf_weight": 1,
    "isolation_hf_weight": 1,
    "input_dim": [224, 224, 3] if mode == "tf" else [3, 224, 224],
    "isolation": True,
    "logit_scale": 1,
    "log_freq": 100,
    "lr": 0.002,
    "max_isolate_classes": min(3, len(class_list)),
    "max_steps": 1000,
    "seed": 0,
    "use_baseline": False,
    "transform": "xl",
    "wandb_api_key": None,
    "wandb_entity": None,
}
```

- **use_alpha** (`bool`): If True, adds an alpha channel to the prototype. This results in the prototype generation process returning semi-transparent prototypes, which allow it to express ambivalence about the values of pixels that don't change the model prediction.
  - *Default*: `False`

- **alpha_mask** (`bool`): If True, applies a mask during prototype generation which encourages the resulting prototypes to be minimal, centered and concentrated. Experimental.
  - *Default*: `False`

- **alpha_only** (`bool`): If True, during the prototype generation process, only an alpha channel is optimised. This results in generation prototypical shapes and textures only, with no colour information.
  - *Default*: `False`

- **baseline_init** (`int` or `str`): How to initialise the input. A sensible option is the mean of your expected input data, if you know it. Use 'r' to initialise with random noise for more varied results with different random seeds.
  - *Default*: `0`

- **diversity_weight** (`int`): When generating multiple prototypes for the same class, we can apply a diversity objective to push for more varied inputs. The higher this number, the harder the optimisation process will push for different inputs. Experimental.
  - *Default*: `0`

- **isolate_classes** (`list`): If you'd like to isolate features for specific classes, rather than the top _n_, specify their indices here, e.g. [2,7,8].
  - *Default*: `None`

- **isolation_lr** (`float`): How much to update the isolation mask at each step during the feature isolation process.
  - *Default*: `0.05`

- **hf_weight** (`int`): How much to penalise high-frequency patterns in the input. If you are generating very blurry and indistinct prototypes, decrease this. If you are getting very noisy prototypes, increase it. This depends on your model architecture and is hard for us to predict, so you might want to experiment. It's a bit like focussing binoculars. Best practice is to start with zero, and gradually increase.
  - *Default*: `1`

- **isolation_hf_weight** (`int`): How much to penalise high-frequency patterns in the feature isolation mask. See hf_weight.
  - *Default*: `1`

- **input_dim** (`list`): The dimensions of the input that your model expects.
  - *Default*: `[224, 224, 3]` if mode is "tf" else `[3, 224, 224]`

- **isolation** (`bool`): Whether to isolate features for entangled classes. Set to False if you want prototypes only.
  - *Default*: `True`

- **log_freq** (`int`): Interval at which to log images.
  - *Default*: `100`

- **lr** (`float`): How much to update the prototype at each step during the prototype generation process. This can be tuned, but in practice is to around 1% of the expected input range. E.g. if your model was trained on images in the range -1 to 1 (prior to any preprocessing function), 0.02 is a good place to start.
  - *Default*: `0.005`

- **max_isolate_classes** (`int`): How many classes to isolate features for, if isolate_classes is not provided.
  - *Default*: `min(3, len(class_list))`

- **max_steps** (`int`): How many steps to run the prototype generation/feature isolation process for. If you get indistinct prototypes or isolations, try increasing this number.
  - *Default*: `1000`

- **seed** (`int`): Random seed for initialisation.
  - *Default*: `0`

- **use_baseline** (`bool`): Whether to generate an equidistant baseline input prior to the prototype generation process. It takes a bit longer, but setting this to True will ensure that all prototypes generated for a model are not biased by input initialisation.
  - *Default*: `False`

- **transform** (`str`): If your model is trained on inputs with non-location-independent features – for example, brain scans, setting this to None will probably result in more sensible prototypes. VERY experimental. You can also experiment with the following options: ['s', 'm', 'l', 'xl'].
  - *Default*: `xl`

- **wandb_api_key** (`str`): Provide your weights and biases API key here to enable logging results directly to your WandB dashboard.
  - *Default*: `None`

- **wandb_entity** (`str`): If logging to WandB, make sure to provide your WandB entity name here.
  - *Default*: `None`

## FAQ

### What is a prototype?

Prototype generation is a global interpretability method. It provides insight into what a model has learned _without_ looking at its performance on test data, by extracting learned features directly from the model itself. This is important, because there's no guarantee that your test data covers all potential failure modes. It's another way of understanding _what_ your model has learned, and helping you to predict how it will behave in deployment, on unseen data.

So what is a prototype? For each class that your model has been trained to predict, we can generate an input that maximises the probability of that output – this is the model's _prototype_ for that class. It's a representation of what the model 'thinks' that class _is_.

For example, if you have a model trained to diagnose cancer from biopsy slides, prototype generation can show you what the model has learned to look for - what it 'thinks' malignant cells look like. This means you can check to see if it's looking for the right stuff, and ensure that it hasn't learned any spurious correlations from its training data that would cause dangerous mistakes in deployment (e.g. looking for lab markings on the slides, rather than at cell morphology).

### What is entanglement?

During the prototype generation process we extract a lot of information from the model, including which other classes _share features_ with the class prototype that we're generating. Depending on your domain, some entanglement may be expected - for example, an animal classifier is likely to have significant entanglement between 'cat' and 'dog', because those classes share (at least) the 'fur' feature. However, entanglement - especially unexpected entanglement, that doesn't make sense in your domain - can also be a very good indicator of where your model is likely to make misclassifications in deployment.

### What is feature isolation?

Feature isolation does what it says on the tin - it isolates which features in the input the model is using to make its prediction. 

We can apply feature isolation in two ways: 
- 1. On a prototype that we've generated, to isolate which features are shared between entangled classes, and so help explain how those classes are entangled; and
- 2. On some input data, to explain individual predictions that your model makes, by isolating the features in the input that correspond to the predicted class (similar to saliency mapping).

So, you can use it to both understand properties of your model as a whole, and to better understand the individual predictions it makes.
