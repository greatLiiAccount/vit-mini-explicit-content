![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rhxwuZq4nbQhPBGDVPc_v.png)

# **vit-mini-explicit-content**

> **vit-mini-explicit-content** is an image classification vision-language model fine-tuned from **vit-base-patch16-224-in21k** for a single-label classification task. It categorizes images based on their explicitness using the **ViTForImageClassification** architecture.

> \[!Note]
> This model is designed to promote safe, respectful, and responsible online spaces. It does **not** generate explicit content; it only classifies images. Misuse may violate platform or regional policies and is strongly discouraged.

> [!Note]
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale : https://arxiv.org/abs/2010.11929, Visual Transformers: Token-based Image Representation and Processing for Computer Vision: https://arxiv.org/pdf/2006.03677

> [!Important]
Note: Explicit, sensual, and pornographic content may appear in the results; however, all of them are considered not safe for work.

```py
Classification Report:
                     precision    recall  f1-score   support

      Anime Picture     0.9077    0.7937    0.8469      5600
Extincing & Sensual     0.9245    0.9717    0.9475      5618
             Hentai     0.8680    0.9391    0.9021      5600
        Pornography     0.9614    0.9544    0.9579      5970
      Safe for Work     0.9235    0.9235    0.9235      6000

           accuracy                         0.9171     28788
          macro avg     0.9170    0.9165    0.9156     28788
       weighted avg     0.9177    0.9171    0.9163     28788
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/VaSWP4-JjXrczImMGufQE.png)

---

The model categorizes images into five classes:

* **Class 0:** Anime Picture
* **Class 1:** Enticing & Sensual
* **Class 2:** Hentai
* **Class 3:** Pornography
* **Class 4:** Safe for Work

# **Run with Transformers**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

## Demo Inference

> [!warning]
Anime Picture

![Screenshot 2025-05-19 at 22-30-24 vit-mini-explicit-content.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/nzHUSO_YN-t2yOMEDT37B.png)

> [!warning]
Extincing & Sensual

![Screenshot 2025-05-19 at 22-30-56 vit-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/78om_bUqjyjyLrrUkWPc2.png)
![Screenshot 2025-05-19 at 22-31-48 vit-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Y6haGPcaoaOj_uM8MX0E7.png)

> [!warning]
Hentai

![Screenshot 2025-05-19 at 22-32-42 vit-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/A48Ow5GllARvz66ZL08Tn.png)

> [!warning]
Pornography

![Screenshot 2025-05-19 at 22-37-31 vit-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/W0CPq8cPb79fpNEqVLIU_.png)

> [!warning]
Safe for Work

![Screenshot 2025-05-19 at 22-27-20 vit-mini-explicit-content.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/R3hnvsYOFh9wA4Y60REKu.png)

---

# **Recommended Use Cases**

* Image moderation pipelines
* Parental and institutional content filters
* Dataset cleansing before training
* Online safety and well-being platforms
* Enhancing search engine filtering

# **Discouraged / Prohibited Use**

* Non-consensual or malicious monitoring
* Automated judgments without human review
* Misrepresentation of moderation systems
* Use in unlawful or unethical surveillance
* Harassment, exploitation, or shaming
