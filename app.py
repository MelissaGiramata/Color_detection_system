import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2

model_filename = 'model.pkl'
with open(model_filename, 'rb') as file:
    knn_model = pickle.load(file)

def preprocess_image(uploaded_image):
    # Convert BytesIO object to file-like object
    image_stream = uploaded_image

    # Load the image using cv2.imread
    cv_image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input expected by the model
    resized_image = cv2.resize(cv_image, (64, 64))

    # Flatten the array and calculate the mean RGB values
    mean_rgb = np.mean(resized_image, axis=(0, 1))

    return mean_rgb

def predict_color_from_image(uploaded_image):
    # Preprocess the image
    mean_rgb = preprocess_image(uploaded_image)

    # Reshape mean_rgb into a 2D array with three features
    mean_rgb_2d = mean_rgb.reshape(1, -1)

    # Predict the color based on the preprocessed image
    prediction = knn_model.predict(mean_rgb_2d)

    return prediction[0]

# Assuming you have a mapping from numerical values to color names
def map_to_color_name(prediction_scalar):
    if prediction_scalar == 0:
        return 'Black'
    elif prediction_scalar == 1:
        return 'Blue'
    elif prediction_scalar == 2:
        return 'Brown'
    elif prediction_scalar == 3:
        return 'Green'
    elif prediction_scalar == 4:
        return 'Grey'
    elif prediction_scalar == 5:
        return 'Orange'
    elif prediction_scalar == 6:
        return 'Red'
    elif prediction_scalar == 7:
        return 'Violet'
    elif prediction_scalar == 8:
        return 'White'
    elif prediction_scalar == 9:
        return 'Yellow'
    else:
        return 'Unknown'

def main():
    # Title with Markdown formatting for blue color
    st.markdown("<h1 style='color: blue;'>Color Prediction App</h1>", unsafe_allow_html=True)

    # Subsection providing additional information to users with smaller font
    st.markdown("<h3 style='font-size: 16px;'>Note: This app was trained to recognize solid colors such as black, blue, brown, green, grey, orange, red, violet, white, and yellow. It might not perform well with images containing multiple colors as it will round them to one color.</h3>", unsafe_allow_html=True)

    # User input for uploading an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Button to trigger color prediction
        if st.button("Predict Color"):
            # Process the image and predict the color
            try:
                prediction = predict_color_from_image(uploaded_image)

                # Convert the NumPy array to a scalar value
                prediction_scalar = prediction.item() if prediction.size == 1 else -1

                # Map numerical value to color name using if statements
                predicted_color_name = map_to_color_name(prediction_scalar)

                # Display the predicted color
                st.subheader("Predicted Color:")
                st.write(predicted_color_name)
            except Exception as e:
                st.error("Error processing the image. Please try another image.")
                st.exception(e)

if __name__ == "__main__":
    main()
