from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Streamlit runs this file as a script and puts src/ on sys.path,
# so a plain import works (a relative ".utils" import would fail).
from utils import (
    build_inference_transform,
    build_model,
    load_checkpoint,
)


CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "resnet18_cifar10.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_model_for_inference(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(checkpoint, device)
    model.eval()
    transform = build_inference_transform(checkpoint)
    return model, checkpoint["class_names"], transform


def classify(image: Image.Image, model, class_names, transform, top_k: int = 3):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    k = min(top_k, len(class_names))
    top_probs, top_idx = probs.topk(k)
    return [(class_names[i], p.item()) for p, i in zip(top_probs, top_idx)]


def main() -> None:
    st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️")
    st.title("CIFAR-10 Image Classifier")
    st.caption(f"ResNet-18 • running on {DEVICE}")

    model, class_names, transform = load_model_for_inference(CHECKPOINT, DEVICE)

    with st.sidebar:
        st.header("Model info")
        st.write("**Device:**", str(DEVICE))
        st.write("**Classes:**", ", ".join(class_names))
        with st.expander("Model architecture"):
            st.code(str(model))

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    results = classify(image, model, class_names, transform)
    best_label, best_prob = results[0]
    st.subheader(f"Prediction: {best_label}")
    st.write(f"Confidence: {best_prob:.1%}")

    st.write("Top predictions")
    for label, prob in results:
        st.progress(prob, text=f"{label}: {prob:.1%}")


main()