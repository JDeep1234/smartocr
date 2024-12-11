import streamlit as st  
from transformers import AutoProcessor, AutoModelForImageClassification  
from PIL import Image  
import re  
from datetime import datetime  
import torch  
import pandas as pd  

# Load the lighter model and processor  
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2")  
processor = AutoProcessor.from_pretrained("google/mobilenet_v2")  

# Streamlit app title  
st.title("Product Information Extractor")  

# Upload image  
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  

if uploaded_file is not None:  
    # Load the image  
    image = Image.open(uploaded_file)  
    st.image(image, caption='Uploaded Image', use_column_width=True)  

    # Define the user message  
    user_message = st.text_area("Enter the text prompt for extraction:",   
                                 "Extract brand name, expiry date, expired status, expected life span in days, and object counts.")  

    if st.button("Extract Information"):  
        # Process the image  
        inputs = processor(images=image, return_tensors="pt")  
        
        # Generate predictions  
        with torch.no_grad():  
            outputs = model(**inputs)  
        
        # Assume we get the output text from the model (this needs to be adapted based on the actual model's output)  
        output_text = "Brand: ExampleBrand, Expiry Date: 12/12/2025, 5 objects"  

        # Adjusted regex patterns for various expiry date formats  
        date_patterns = [  
            r'\b(\d{2}/\d{2}/\d{4})\b',  # MM/DD/YYYY  
            r'\b(\d{2}-\d{2}-\d{4})\b',  # MM-DD-YYYY  
            r'\b(\d{2}/\d{2}/\d{2})\b',  # MM/DD/YY  
            r'\b(\d{2}-\d{2}-\d{2})\b',  # MM-DD-YY  
            r'\b(\d{2} \w+ \d{4})\b',    # DD Month YYYY  
            r'\b(\d{2} \d{2} \d{4})\b'   # DD MM YYYY  
        ]  

        # Extract expiry date  
        expiry_date = None  
        for pattern in date_patterns:  
            match = re.findall(pattern, output_text)  
            if match:  
                expiry_date = match[0]  
                break  

        # Extract brand name  
        brand_name = None  
        brand_patterns = [  
            r"brand[\s:]*([A-Za-z0-9\s]+)",  
            r"([A-Za-z]+)[\s]+brand"  
        ]  
        for pattern in brand_patterns:  
            match = re.findall(pattern, output_text)  
            if match:  
                brand_name = match[0]  
                break  

        # Determine if the product is expired  
        expired = None  
        if expiry_date:  
            try:  
                expiry_date = datetime.strptime(expiry_date, "%d/%m/%Y")  
                current_date = datetime.now()  
                expired = expiry_date < current_date  
            except ValueError:  
                st.error(f"Could not parse the expiry date format: {expiry_date}")  

        # Calculate expected life span if not expired  
        life_span_days = None  
        if not expired and expiry_date:  
            life_span_days = (expiry_date - current_date).days  

        # Object count extraction  
        object_count = None  
        count_pattern = r"(\d+)\s*objects?|(\d+)\s*items?"  
        count_match = re.findall(count_pattern, output_text)  
        if count_match:  
            object_count = int(count_match[0][0])  

        # Prepare data for display in a table  
        data = {  
            "Attribute": ["Brand Name", "Expiry Date", "Expired", "Expected Life Span (Days)", "Object Count"],  
            "Value": [  
                brand_name if brand_name else 'Not found',  
                expiry_date.strftime('%d/%m/%Y') if expiry_date else 'Not found',  
                'Yes' if expired else 'No' if expired is not None else 'N/A',  
                life_span_days if life_span_days is not None else 'N/A',  
                object_count if object_count is not None else 'Not found'  
            ]  
        }  

        # Create a DataFrame and display it as a table  
        df = pd.DataFrame(data)  
        st.table(df)
