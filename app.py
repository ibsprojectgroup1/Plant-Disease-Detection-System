

import streamlit as st

from prediction import (
    load_model,
    load_leaf_validator,
    predict_leaf
)


# =====================================
# 1. Streamlit Page Configuration
# =====================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection System")

st.write(
    "Upload a cassava or maize leaf image to identify "
    "possible diseases using deep learning."
)

st.info(
    "Accuracy Notice: Cassava disease predictions may be less reliable "
    "than expected. Results are intended for preliminary guidance and "
    "should be verified with a plant disease expert."
)


# =====================================
# 2. Load and Cache Models
# =====================================

@st.cache_resource
def get_model(model_name):
    return load_model(model_name)


@st.cache_resource
def get_leaf_validator():
    return load_leaf_validator()


# =====================================
# 3. Model Selection
# =====================================

model_type = st.selectbox(
    "Select Classification Model",
    ["ResNet50", "EfficientNetB0"]
)


# =====================================
# 4. Image Upload
# =====================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Disease"):

        try:

            with st.spinner("Analyzing image..."):

                # Load selected disease model
                model = get_model(model_type)

                # Load MobileNetV2 validator
                validator = get_leaf_validator()

                # Run the complete prediction pipeline
                uploaded_file.seek(0)

                result = predict_leaf(
                    uploaded_file,
                    model_type=model_type,
                    model=model,
                    validator=validator
                )

            # =================================
            # 5. Display Prediction Results
            # =================================

            if not result["is_leaf"]:

                st.error(result["message"])

                st.info(
                    "Please upload a valid cassava or maize leaf image."
                )

            else:

                st.success("Leaf image successfully validated!")

                st.subheader("Disease Prediction")

                st.write(
                    f"**Predicted Class:** "
                    f"{result['predicted_class']}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{result['confidence'] * 100:.2f}%"
                )

                # Display class probabilities
                st.subheader("Class Probabilities")

                st.bar_chart(
                    result["all_probabilities"]
                )

        except Exception as e:

            st.error(
                f"An error occurred during prediction: {e}"
            )
