import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
from datetime import datetime

# Load the BLIP model and processor
MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

def extract_info(text):
    # Regex patterns for extraction
    date_patterns = [
        r'\b(\d{2}/\d{2}/\d{4})\b',
        r'\b(\d{2}-\d{2}-\d{4})\b',
        r'\b(\d{2}/\d{2}/\d{2})\b',
        r'\b(\d{2}-\d{2}-\d{2})\b',
        r'\b(\d{2} \w+ \d{4})\b',
        r'\b(\d{2} \d{2} \d{4})\b'
    ]

    brand_patterns = [
        r"brand[\s:]*([A-Za-z0-9\s]+)",
        r"([A-Za-z]+)[\s]+brand"
    ]

    count_pattern = r"(\d+)\s*(objects?|items?)"

    # Extract expiry date
    expiry_date = None
    for pattern in date_patterns:
        match = re.findall(pattern, text)
        if match:
            expiry_date = match[0]
            break

    # Extract brand name
    brand_name = None
    for pattern in brand_patterns:
        match = re.findall(pattern, text)
        if match:
            brand_name = match[0]
            break

    # Determine if expired
    expired = None
    if expiry_date:
        try:
            expiry_date_obj = datetime.strptime(expiry_date, "%d/%m/%Y")
            expired = expiry_date_obj < datetime.now()
        except ValueError:
            pass

    # Calculate expected life span
    life_span_days = None
    if expiry_date and not expired:
        life_span_days = (expiry_date_obj - datetime.now()).days

    # Extract object count
    object_count = None
    count_match = re.findall(count_pattern, text)
    if count_match:
        object_count = int(count_match[0][0])

    return brand_name, expiry_date, expired, life_span_days, object_count

def generate_output(image, query):
    inputs = processor(images=image, text=query, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Streamlit app UI
st.title("Enhanced Streamlit App with BLIP for Information Extraction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Enter your query", "Extract brand name, expiry date, expired status, expected life span in days, and object counts.")

if uploaded_file and text_input:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")

    # Process the image and query with BLIP
    generated_text = generate_output(image, text_input)

    # Extract information from generated text
    brand_name, expiry_date, expired, life_span_days, object_count = extract_info(generated_text)

    st.subheader("Extracted Information")
    st.write(f"**Brand Name:** {brand_name or 'Not found'}")
    st.write(f"**Expiry Date:** {expiry_date or 'Not found'}")
    st.write(f"**Expired:** {'Yes' if expired else 'No' if expired is not None else 'Not determined'}")
    st.write(f"**Expected Life Span in Days:** {life_span_days if life_span_days is not None else 'N/A'}")
    st.write(f"**Object Count:** {object_count if object_count is not None else 'Not found'}")
