
import streamlit as st
from prediction import load_model, predict_leaf

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Application title
st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a cassava or maize leaf image "
    "to identify its possible disease."
)

# Model selection
model_type = st.selectbox(
    "Select Model",
    ["ResNet50", "EfficientNetB0"]
)

# Load and cache model
@st.cache_resource
def get_model(model_name):
    return load_model(model_name)

# Image upload
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# Display uploaded image
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("Predict Disease"):

        with st.spinner("Analyzing image..."):

            model = get_model(model_type)

            uploaded_file.seek(0)

            result = predict_leaf(
                uploaded_file,
                model_type=model_type,
                model=model
            )

        st.success("Prediction completed!")

        st.subheader("Prediction Result")

        st.write("Predicted Class:", result["predicted_class"])

        st.metric(
            "Confidence",
            f'{result["confidence"] * 100:.2f}%'
        )

        st.subheader("Class Probabilities")

        st.bar_chart(result["all_probabilities"])
