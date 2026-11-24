from hashlib import sha256
from io import BytesIO

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models
import pandas as pd

st.set_page_config(
    page_title="Image Classifier | PyTorch",
    page_icon="🖼️",
    layout="wide",
)

MODEL_OPTIONS = {
    "ResNet-18": {
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "description": "A compact residual network that is a good starting point.",
    },
}

DEVICE = torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device_name: str):
    config = MODEL_OPTIONS[model_name]
    weights = config["weights"]
    model = config["builder"](weights=weights)
    model.to(device_name)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def classify_image(image: Image.Image, model_names: list[str], top_k: int):
    predictions = {}
    errors = []

    for model_name in model_names:
        try:
            model, preprocess, labels = load_model(model_name, str(DEVICE))
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor)[0], dim=0)

            scores, class_ids = torch.topk(probabilities, top_k)
            predictions[model_name] = [
                {"Rank": rank, "Label": labels[class_id], "Confidence": score}
                for rank, (score, class_id) in enumerate(
                    zip(scores.tolist(), class_ids.tolist()), start=1
                )
            ]
        except (OSError, RuntimeError) as exc:
            errors.append(f"{model_name}: {exc}")

    return predictions, errors


st.title("Interactive Image Classifier")
st.write(
    "Classify an uploaded image or webcam photo into ImageNet's 1,000 categories "
    "using a pre-trained PyTorch ResNet18 model running on CPU."
)

with st.sidebar:
    st.header("Classification settings")
    with st.form("classification_settings"):
        selected_models = st.multiselect(
            "Models to compare",
            options=list(MODEL_OPTIONS),
            default=["ResNet-18"],
            help="Uses the pre-trained ResNet18 model for image classification.",
        )
        top_k = st.slider("Predictions per model", min_value=1, max_value=10, value=5)
        run_classification = st.form_submit_button(
            "Classify image",
            type="primary",
            use_container_width=True,
        )

    st.caption(f"Inference device: **{DEVICE.type.upper()}**")
    with st.expander("About the models"):
        for name, config in MODEL_OPTIONS.items():
            st.markdown(f"**{name}** — {config['description']}")
        st.caption(
            "Confidence scores are each model's ImageNet class probabilities; "
            "they are not calibrated for direct accuracy comparisons."
        )

image_source = st.radio(
    "Choose an image source",
    ["Upload an image", "Take a webcam photo"],
    horizontal=True,
)

if image_source == "Upload an image":
    image_data = st.file_uploader(
        "Choose a JPG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    image_data = st.camera_input("Take a photo")

if image_data is None:
    st.info("Upload an image or take a webcam photo to get started.")
    st.stop()

image_bytes = image_data.getvalue()
try:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
except (UnidentifiedImageError, OSError) as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

st.image(image, caption="Image to classify", width="stretch")
image_signature = sha256(image_bytes).hexdigest()

if run_classification:
    if not selected_models:
        st.error("Select at least one model before starting classification.")
    else:
        with st.spinner("Running the selected models..."):
            predictions, errors = classify_image(image, selected_models, top_k)
        st.session_state["last_classification"] = {
            "image_signature": image_signature,
            "models": selected_models,
            "top_k": top_k,
            "predictions": predictions,
            "errors": errors,
        }

result = st.session_state.get("last_classification")
if result is not None:
    settings_match = (
        result["image_signature"] == image_signature
        and result["models"] == selected_models
        and result["top_k"] == top_k
    )
    if settings_match:
        for error in result["errors"]:
            st.error(
                "Could not run a model. Check your connection and available memory. "
                f"{error}"
            )

        predictions = result["predictions"]
        if predictions:
            
            for model_name, model_predictions in predictions.items():
                best_prediction = model_predictions[0]
                st.metric(
                    "Top Prediction",
                    best_prediction["Label"],
                    f"{best_prediction['Confidence']:.1%}",
                )

                # Step 8: Display Top-5 predicted classes and probabilities using Pandas
                st.subheader("Top-5 Predicted Classes")
                df_top5 = pd.DataFrame(model_predictions)
                st.dataframe(df_top5, hide_index=True, use_container_width=True)

                # Step 9: Visualize the Top-5 prediction probabilities using a bar chart
                st.subheader("Prediction Probabilities Bar Chart")
                st.bar_chart(df_top5.set_index("Label")["Confidence"])

        elif not result["errors"]:
            st.warning("No predictions were returned. Try selecting another model.")
    else:
        st.info("Settings changed. Click **Classify image** to run the selected models.")
