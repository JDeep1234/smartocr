import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
import torch

# Load the BLIP model and processor
MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

def extract_basic_info(text):
    # Extract brand name (simplified heuristic)
    brand_name = None
    if "brand" in text.lower():
        brand_name = text.split("brand")[-1].split()[0]

    # Extract expiry date (simple format matching)
    expiry_date = None
    for pattern in [r'\b(\d{2}/\d{2}/\d{4})\b', r'\b(\d{2}-\d{2}-\d{4})\b']:
        match = re.search(pattern, text)
        if match:
            expiry_date = match.group(0)
            break

    # Extract object count
    object_count = None
    if "objects" in text.lower() or "items" in text.lower():
        try:
            object_count = int([int(s) for s in text.split() if s.isdigit()][0])
        except (ValueError, IndexError):
            pass

    return brand_name, expiry_date, object_count

def generate_output(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Streamlit app UI
st.title("Basic Brand, Expiry Date, and Object Count Extractor")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")

    # Process the image with BLIP
    generated_text = generate_output(image)

    # Extract basic information
    brand_name, expiry_date, object_count = extract_basic_info(generated_text)

    st.subheader("Extracted Information")
    st.write(f"**Brand Name:** {brand_name or 'Not found'}")
    st.write(f"**Expiry Date:** {expiry_date or 'Not found'}")
    st.write(f"**Object Count:** {object_count if object_count is not None else 'Not found'}")
