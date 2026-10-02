# Open AI CLIP vs imagenette

## The Model

This notebook analyzed OpenAI's [CLIP](https://github.com/openai/CLIP/) computer vision model. This model makes an interesting claim: it can categorize images with high accuracy, and you can define those classes when you script it for your task. This extends broadly to include classes it wasn't explicitly trained on. This fits the core definition of what AI excels at: 'a program's ability to compute outside what it was explicitly coded for.'

**Downloading the model**:
```bash
$ python download_model.py
```

## The Dataset

For evaluation, I chose a pre-labeled dataset, [imagenette](https://www.tensorflow.org/datasets/catalog/imagenette). The dataset contains ~13,000 images that represent ten distinct classes:

| Class | Description |
|-------|-------------|
| 0 | Tench (fish)    |
| 1 | English Springer|
| 2 | Casette player  |
| 3 | Chainsaw        |
| 4 | Church          |
| 5 | French Horn     |
| 6 | Garbage truck   |
| 7 | Gas pump        |
| 8 | Golf ball       |
| 9 | Parachute       |

## Results

Overall, the model performed very well in every category evaluated. [analysis](analysis.ipynb) indicates that this model should perform well whenever the classes of images being evaluated fall into distinct categories and the subject of the image is the goal, not the 'story' of what is happening in the image. This is because the model was designed as a classifier and not with any 'reasoning' ability.

**Use Cases**: Broad computer vision tasks where a lightweight model is needed in order to categorize an image into a set list of choices that's provided at runtime. Under the right conditions, this model is a low-effort choice that is light enough to be usable in CPU-only and potentially embedded applications.

# Example - Single-image Classification at runtime

## Example 1
In this example, the model can identify a dog in an image given the choices `dog, horse, and cow`

```bash
$ python clip_file.py ./static/example.jpeg dog horse cow

# output:
Loading CLIP locally from local_clip_model...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:00<00:00, 8605.34it/s]
FILE: ./static/example.jpeg
        PREDICTION: dog CONFIDENCE: 0.9687
```

## Example 2
Conversely, if `animal` is a more general option, the model determines turkey is an animal, though `turkey` isn't explicitly given as a parameter.

```bash
$ python clip_file.py ./static/example2.jpeg dog horse cow animal

# output:
Loading CLIP locally from local_clip_model...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:00<00:00, 7590.71it/s]
FILE: ./static/example2.jpeg
        PREDICTION: animal CONFIDENCE: 0.9803
```

Lastly, given a photo of a computer and the same parameters as before, it gives a very low prediction, which could be filtered out

```bash
$ python clip_file.py ./static/example3.jpeg dog horse cow animal

# output:
Loading CLIP locally from local_clip_model...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:00<00:00, 1821.94it/s]
FILE: ./static/example3.jpeg
        Unknown Object!!
```