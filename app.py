import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import re
from datetime import datetime
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the lightweight model and processor
model_name = "Salesforce/blip-image-captioning-base"  # Lightweight vision-language model
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Streamlit app title
st.title("Product Metadata Extractor App")
st.write("Upload an image of a product and extract metadata such as brand name, expiry date, and more!")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # User prompt
    user_prompt = st.text_input(
        "Enter your query:",
        "Extract brand name, expiry date, expired status, expected life span in days, and object counts."
    )

    if st.button("Process Image"):
        # Process the image using BLIP
        inputs = processor(images=image, text=user_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        decoded_output = processor.decode(outputs[0], skip_special_tokens=True)

        # Display raw output
        st.subheader("Raw Output")
        st.write(decoded_output)

        # Process output
        date_patterns = [
            r'\b(\d{2}/\d{2}/\d{4})\b',
            r'\b(\d{2}-\d{2}-\d{4})\b'
        ]

        expiry_date = None
        for pattern in date_patterns:
            match = re.search(pattern, decoded_output)
            if match:
                expiry_date = match.group(0)
                break

        brand_pattern = r"brand[\s:]*([A-Za-z0-9\s]+)"
        brand_match = re.search(brand_pattern, decoded_output, re.IGNORECASE)
        brand_name = brand_match.group(1).strip() if brand_match else "Not found"

        count_pattern = r"(\d+)\s*objects?"
        count_match = re.search(count_pattern, decoded_output)
        object_count = int(count_match.group(1)) if count_match else "Not found"

        # Expiry check
        expired = None
        life_span_days = None
        if expiry_date:
            try:
                expiry_date_dt = datetime.strptime(expiry_date, "%d/%m/%Y")
                current_date = datetime.now()

                expired = expiry_date_dt < current_date
                if not expired:
                    life_span_days = (expiry_date_dt - current_date).days
            except ValueError:
                expired = "Invalid date format"

        # Display results
        st.subheader("Extracted Metadata")
        st.write(f"**Brand Name:** {brand_name}")
        st.write(f"**Expiry Date:** {expiry_date if expiry_date else 'Not found'}")
        st.write(f"**Expired:** {'Yes' if expired else 'No' if expired is not None else 'Unknown'}")
        st.write(f"**Expected Life Span in Days:** {life_span_days if life_span_days is not None else 'N/A'}")
        st.write(f"**Object Count:** {object_count}")
