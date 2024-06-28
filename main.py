import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


def homepage():
    st.title("Plant Disease Prediction App")
    st.write("""
    Welcome to the Plant Disease Prediction App! 
    This app predicts the disease affecting plants based on input features.
    """)

    st.subheader("Features:")
    st.write("- Upload an image of a plant leaf for disease prediction.")
    st.write("- View information about the prediction model.")
    st.write("- Learn about common plant diseases.")

    st.subheader("Explore Leaf Diseases:")
    plant_leaves = [
        "Apple leaf",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pepper",
        "Potato",
        "Raspberry",
        "Soybean",
        "Strawberry",
        "Tomato"
    ]
    selected_plant = st.selectbox("Select a plant leaf", plant_leaves)

    disease_info = {
        "Apple leaf": ["Early Blight", "Scab", "Rust", "Powdery Mildew"],
        "Blueberry": ["Black Rot", "Haunglongbing", "Bacterial Spot"],
        "Cherry": ["Late Blight", "Leaf Mold", "Mosaic Virus"],
        "Corn": ["Common Rust", "Northern Leaf Blight", "Gray Leaf Spot"],
        "Grape": ["Black Rot", "Downy Mildew", "Powdery Mildew"],
        "Orange": ["Citrus Canker", "Citrus Black Spot", "Citrus Greening"],
        "Peach": ["Bacterial Spot", "Peach Leaf Curl", "Brown Rot"],
        "Pepper": ["Bacterial Spot", "Powdery Mildew", "Anthracnose"],
        "Potato": ["Late Blight", "Early Blight", "Blackleg"],
        "Raspberry": ["Anthracnose", "Powdery Mildew", "Verticillium Wilt"],
        "Soybean": ["Soybean Cyst Nematode", "Brown Spot", "White Mold"],
        "Strawberry": ["Anthracnose", "Leaf Scorch", "Powdery Mildew"],
        "Tomato": ["Early Blight", "Late Blight", "Septoria Leaf Spot"]
    }

    if selected_plant in disease_info:
        selected_diseases = disease_info[selected_plant]
        selected_disease = st.selectbox("This leaf can have majorly following diseases", selected_diseases)
    #     st.write(f"Information about {selected_disease} goes here...")


def prediction_page():
    
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"Model/plant_disease_model draft 2.h5"
    # model_weights_path = f"{working_dir}/trained_model/weights.h5"
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)

    # model.load_weights(model_weights_path)
    # loading the class names
    class_indices = json.load(open(f"Model/Class_indices.json"))

    
    # Function to Load and Preprocess the Image using Pillow
    def load_and_preprocess_image(image_path, target_size=(224, 224)):
        # Load the image
        img = Image.open(image_path)
        # Resize the image
        img = img.resize(target_size)
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Scale the image values to [0, 1]
        img_array = img_array.astype('float32')
        return img_array


    # Function to Predict the Class of an Image
    def predict_image_class(model, image_path, class_indices):
        preprocessed_img = load_and_preprocess_image(image_path)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]
        return predicted_class_name

    # Load sample images
    sample_images = ["soyabean.jpg", 'Frogeye.jpg','apple_black_rot.jpg']
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# Display sample images in a single row
    st.write("Or choose from the following sample images for testing:")
    img1 , img2 , img3 = st.columns(3)
    
    with img1:
        sample_img1 = Image.open(sample_images[0])
        st.image(sample_img1, width=200)
        if st.button(f'{sample_images[0]}'):
            prediction = predict_image_class(model, sample_images[0], class_indices)
            st.success(f'Prediction:,{str(prediction)}')
            
    with img2:
        sample_img2 = Image.open(sample_images[1])
        st.image(sample_img2, width=200)
        if st.button(f'{sample_images[1]}'):
            prediction = predict_image_class(model, sample_images[1], class_indices)
            st.success(f'Prediction:,{str(prediction)}')
    
    with img3:
        sample_img3 = Image.open(sample_images[2])
        st.image(sample_img3, width=200)
        if st.button(f'{sample_images[2]}'):
            prediction = predict_image_class(model, sample_images[2] ,class_indices)
            st.success(f'Prediction:,{str(prediction)}')
    
    

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)
        
        
        with col2:
            if st.button('Classify'):
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction:,{str(prediction)}')


def about_page():
    st.title("About")
    st.write(create_about_section())

    st.subheader("Contact Form")
    name = st.text_input("Name", "")
    email = st.text_input("Email", "")
    message = st.text_area("Message", "")
    if st.button("Send"):
        # Add functionality to send email or store data in database
        st.success("Message sent successfully!")

    st.subheader("Connect with Me")
    st.write("Feel free to reach out to me on LinkedIn or GitHub:")
    linkedin_url = "https://www.linkedin.com/in/pushp-raj-gour"
    github_url = "https://github.com/Pushpraj-Gour"

    linkedin_icon = "<a href='%s' target='_blank'><img src='https://image.flaticon.com/icons/png/512/174/174857.png' width='30' style='margin-right: 10px'></a>" % linkedin_url
    github_icon = "<a href='%s' target='_blank'><img src='https://image.flaticon.com/icons/png/512/25/25231.png' width='30'></a>" % github_url

    st.markdown("<div style='display:flex;align-items:center'>" + linkedin_icon + github_icon + "</div>", unsafe_allow_html=True)
    
def create_about_section():
    about_content = """
    ## Plant Disease Prediction Model

    ### Technical Details

    The plant disease prediction model utilizes a custom convolutional neural network (CNN) architecture tailored for image classification tasks. Various techniques, including data augmentation and regularization methods such as dropout, have been implemented to enhance model performance and prevent overfitting.

    ### Training Data

    The model was trained on a diverse dataset of over 50,000 plant images sourced from Kaggle, covering a wide range of plant species and disease categories. Augmentation strategies such as rotation, scaling, and flipping were applied to enrich the dataset and improve model robustness.

    ### Model Performance

    Throughout the training phase, close monitoring of model performance was conducted, evaluating accuracy and loss on both training and validation datasets. Leveraging data generators for efficient image preprocessing and batching ensured smooth training and satisfactory results in disease classification.

    ### Deployment Information

    The plant disease prediction model is now available as a user-friendly web application built with Streamlit. Users can conveniently upload images of diseased plants via the interface, and the model promptly analyzes the images, providing predictions on disease presence.

    ### About

    Hi, I'm Pushpraj Gour, the sole developer behind this plant disease prediction app. With a passion for machine learning and agriculture, I embarked on this project to address agricultural challenges and promote sustainable farming practices through technology.

    ### Contact Information

    For inquiries, feedback, or collaboration opportunities, please reach out via:

    - Email: rajrjpushp@gmail.com
    - LinkedIn: https://www.linkedin.com/in/pushp-raj-gour
    - GitHub: https://github.com/Pushpraj-Gour

    Thank you for using the plant disease prediction app!
    """
    return about_content
    
def main():
    pages = {
        "Home": homepage,
        "Prediction": prediction_page,
        "About": about_page
    }

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to", list(pages.keys()), key="navigation")

    pages[selected_page]()

if __name__ == "__main__":
    main()

