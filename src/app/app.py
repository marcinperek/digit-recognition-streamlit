import numpy as np
import streamlit as st
import torch

from sklearn.datasets import load_digits

from ml.DigitRecognizer import DigitRecognizer


st.title("Digit Recognizer")

@st.cache_resource
def load_model(path):
    device = torch.device('cpu')  # Ensure CPU is used
    model_data = torch.load(path, map_location=device)  # Load model on CPU
    model = DigitRecognizer()
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)  # Move model to CPU
    model.eval()
    return model


uploaded_image = st.file_uploader("Upload a 8x8 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    from PIL import Image

    image = Image.open(uploaded_image).convert("L").resize((8, 8))
    st.image(image, caption="Uploaded Image", use_column_width=False)

    image_array = np.array(image).astype(np.float32)
    image_array = (image_array / 255.0) * 16
    image_array = image_array.flatten()

    print(image_array)

    model = load_model("models/best_20251218_173936.pth")

    with torch.no_grad():
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).to('cpu')  # Ensure tensor is on CPU
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    st.write(f"Predicted Digit: **{predicted_class}**")