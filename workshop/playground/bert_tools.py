import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("train_images/snivy.jpeg")).unsqueeze(0).to(device) # CLIP.png为本文中图一，即CLIP的流程图
text = clip.tokenize(["photo of 1 cat and 1 dog", "photo of 1 cat and 2 dogs", "photo of 2 cats and 1 dog"]).to(device) # 将这三句话向量化

print(model.visual)
sys.exit()

with torch.no_grad():
    # image_features = model.encode_image(image) # 将图片进行编码
    # text_features = model.encode_text(text)    # 将文本进行编码
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
