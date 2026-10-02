from transformers import CLIPProcessor, CLIPModel

model_id = 'openai/clip-vit-base-patch32'
local_dir = './local_clip_model'

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

model.save_pretrained(local_dir)
processor.save_pretrained(local_dir)

print(f'CLIP model saved to {local_dir}')