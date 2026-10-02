import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

model_id = 'openai/clip-vit-base-patch32'
device = 'cuda'

print(f'Loading {model_id} on {device}...')
model = CLIPModel.from_pretrained(model_id).to(device)
processor=CLIPProcessor.from_pretrained(model_id)

classes = ['tench', 'english springer', 'casette player', 'chainsaw', 'church',
           'french horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

# Model reportedly does better if it has a short description:
text_labels = [f'a photo of a {c}' for c in classes]

image_dir = './images'
image_files = os.listdir(image_dir)

# Holds results:
inference_results = []

for filename in image_files:
    image_path = os.path.join(image_dir, filename)

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

        print(f'FILE: {filename}')
        print(f'\tPREDICTION: {classes[predicted_idx]} (INDEX: {predicted_idx}) CONFIDENCE: {round(confidence, 4)}')

        # Append inference_results:
        inference_results.append({
            'filename': filename,
            'prediction': predicted_idx,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print(f"Failed to process {filename}: {str(e)}")

print('preview of inference results:')
print(inference_results[0:10])

# Convert inference_results to df:
inference_df = pd.DataFrame(inference_results)

# load images.csv so it can be appended:
df = pd.read_csv('images.csv')

final_df = df.merge(inference_df, on='filename', how='left')

print('Head of final dataframe:')
print(final_df.head())

final_df.to_csv('images_with_predictions.csv', index=False)