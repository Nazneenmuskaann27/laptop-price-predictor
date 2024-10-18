import pandas as pd
import random

# Define the properties and their possible values
brands = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Microsoft']
processors = ['Intel i3', 'Intel i5', 'Intel i7', 'AMD Ryzen 5', 'AMD Ryzen 7']
ram_sizes = [4, 8, 16, 32]  # in GB
storage_types = ['SSD', 'HDD']
storage_sizes = [256, 512, 1024, 2048]  # in GB
screen_sizes = [13.3, 14.0, 15.6, 17.3]  # in inches
prices = [500, 600, 750, 850, 1000, 1200, 1500, 2000]  # in USD

# Generate 2000 rows of random laptop data
data = {
    'Brand': [random.choice(brands) for _ in range(2000)],
    'Model Name': [f'Model-{random.randint(100, 999)}' for _ in range(2000)],
    'Processor': [random.choice(processors) for _ in range(2000)],
    'RAM Size': [random.choice(ram_sizes) for _ in range(2000)],
    'Storage Type': [random.choice(storage_types) for _ in range(2000)],
    'Storage Size': [random.choice(storage_sizes) for _ in range(2000)],
    'Screen Size': [random.choice(screen_sizes) for _ in range(2000)],
    'Price': [random.choice(prices) for _ in range(2000)],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv('laptop_price_data.csv', index=False)
import streamlit as st
def set_background_image(image_url):
    # Apply custom CSS to set the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-position: top;
        background-image: url(%s);
        background-size: cover;
    }

    @media (max-width: 768px) {
        /* Adjust background size for mobile devices */
        .stApp {
            background-position: top;
            background-size: contain;
            background-repeat: no-repeat;
        }
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    # Set the background image URL
    background_image_url ="https://img.freepik.com/free-vector/realistic-3d-advert-with-laptop_79603-1257.jpg"
    # Set the background image
    set_background_image(background_image_url)

    custom_css = """
       <style>
       body {
           background-color: #4699d4;
           color: #ffffff;
           font-family: Arial, sans-serif;
       }
       select {
           background-color: #000000 !important; /* Black background for select box */
           color: #ffffff !important; /* White text within select box */
       }
       label {
           color: #ffffff !important; /* White color for select box label */
           font-size: 20px; /* Adjust font size for labels */
           font-weight: bold; /* Make labels bold */
       }
       h1, h2, h3 {
           font-weight: bold; /* Make headings bold */
           font-size: 24px; /* Adjust heading font size */
       }
       .whatsapp-stats {
           font-size: 20px; /* Adjust font size for WhatsApp stats */
           font-weight: bold; /* Make WhatsApp stats bold */
           color: #333333; /* Dark color for WhatsApp stats */
       }
       .stBlock {
           border-right: 2px solid #ffffff; /* Draw a vertical line */
           padding-right: 10px;
           margin-right: 10px;
       }
       .stBlock .stFileUploader {
           border: 1px solid #ffffff; /* Add a border around the file uploader */
           padding: 10px;
           margin: 10px;
       }
       </style>
       """

    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import streamlit as st
import pickle

# Load dataset
data = pd.read_csv('laptop_price_data.csv')

# Preprocessing
le = LabelEncoder()
data['Brand'] = le.fit_transform(data['Brand'])
data['Processor'] = le.fit_transform(data['Processor'])
data['Storage Type'] = le.fit_transform(data['Storage Type'])
data = data.drop('Model Name', axis=1)
# Features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the model
with open('laptop_price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the LabelEncoder for later use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model trained and saved!")
import streamlit as st
import pickle
import numpy as np

# Load the trained model and LabelEncoder
with open('laptop_price_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Function to handle unseen labels
def handle_unseen_label(encoder, label, default='other'):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        # Assign a default value for unseen labels
        if default not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, default)
        return encoder.transform([default])[0]

# Streamlit App Layout
st.title("Laptop Price Predictor")

# Input Fields
brand = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Microsoft'])
processor = st.selectbox("Processor", ['Intel i3', 'Intel i5', 'Intel i7', 'AMD Ryzen 5', 'AMD Ryzen 7'])
ram_size = st.slider("RAM Size (GB)", 4, 32, 8)
storage_type = st.selectbox("Storage Type", ['SSD', 'HDD'])
storage_size = st.selectbox("Storage Size (GB)", [256, 512, 1024, 2048])
screen_size = st.selectbox("Screen Size (inches)", [13.3, 14.0, 15.6, 17.3])

# Predict Button
if st.button("Predict Price"):
    # Encode categorical variables, handle unseen labels
    brand_encoded = handle_unseen_label(le, brand)
    processor_encoded = handle_unseen_label(le, processor)
    storage_type_encoded = handle_unseen_label(le, storage_type)

    # Prepare input for prediction
    input_data = np.array([[brand_encoded, processor_encoded, ram_size, storage_type_encoded, storage_size, screen_size]])

    # Predict price
    predicted_price = model.predict(input_data)

    # Display the result
    st.success(f"The predicted price for the laptop is: ${predicted_price[0]:,.2f}")


