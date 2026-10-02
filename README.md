# Open AI CLIP vs imagenette

## The Model

This notebook analyzed OpenAI's [CLIP](https://github.com/openai/CLIP/) computer vision model. This model makes an interesting claim: it can categorize images with high accuracy, and you can define those classes when you script it for your task. This extends broadly to include classes it wasn't explicitly trained on. This fits the core definition of what AI excels at: 'a program's ability to compute outside what it was explicitly coded for.'

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