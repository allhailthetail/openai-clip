import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import argparse
from pathlib import Path
import os
import sys

# Parse runtime arguments:
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('parameters', nargs='+')
args = parser.parse_args()

# Get cwd:
wkdir = os.getcwd()

# Check if file exists:
def file_exists(filename):
    try:
        return Path(filename).is_file()
    except (OSError, ValueError) as e:
        print(f'Error opening file: {e}')
        return False

# Returns search parameters as type: list
def get_parameters(parameters):
    try:
        return list(parameters)
    except:
        sys.exit('Exit: Search parameters are not a list!')

# Check if file actually exists:
file_exists(args.filename)
# Full path to file to be analyzed:
image_path = args.filename
classes = get_parameters(args.parameters)

# Now begin loading the model:
model_dir = 'local_clip_model'
model_id = 'openai/clip-vit-base-patch32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading CLIP locally from {model_dir}...')
model = CLIPModel.from_pretrained(model_dir).to(device)
processor=CLIPProcessor.from_pretrained(model_dir)

# Model reportedly does better if it has a short description:
text_labels = [f'a photo of a {c}' for c in classes]

try:
    image=Image.open(image_path).convert('RGB')

    inputs = processor(
        text=text_labels,
        images=image,
        return_tensors='pt',
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image

    probs = logits_per_image.softmax(dim=1)

    predicted_idx = probs.argmax().item()
    confidence = probs[0, predicted_idx].item()

    # Whatever our threshold for giving a determination is:
    threshold = 0.8

    print(f'FILE: {image_path}')
    if confidence > threshold:
        print(f'\tPREDICTION: {classes[predicted_idx]} CONFIDENCE: {round(confidence, 4)}')
    else:
        print('\tUnknown Object!!')

except Exception as e:
    print(f"Failed to process {image_path}: {str(e)}")