import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/vit-mini-explicit-content"  # Updated model path
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Updated label mapping
labels = {
    "0": "Anime Picture",
    "1": "Enticing & Sensual",
    "2": "Hentai",
    "3": "Pornography",
    "4": "Safe for Work"
}

def explicit_content_detection(image):
    """Predicts the type of content in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=explicit_content_detection,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="vit-mini-explicit-content",
    description="Upload an image to classify whether it is anime, enticing & sensual, hentai, pornographic, or safe for work."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
