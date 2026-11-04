
import streamlit as st

from prediction import load_models, predict_leaf


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
    "Predictions are not guaranteed to be 100% accurate. "
    "Results should be treated as guidance and not as a definite diagnosis."
)


# =====================================
# 2. Load and Cache Models
# =====================================

@st.cache_resource
def get_models():
    return load_models()


# =====================================
# 3. Crop Selection
# =====================================

crop_type = st.selectbox(
    "Select Crop Type",
    ["Cassava", "Maize"]
)

selected_crop = crop_type.lower()


# =====================================
# 4. Image Upload
# =====================================

uploaded_file = st.file_uploader(
    f"Upload a {selected_crop} leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Disease", type="primary"):

        try:

            with st.spinner("Analyzing image..."):

                # Load all five models
                models = get_models()

                uploaded_file.seek(0)

                # Run the complete prediction pipeline
                result = predict_leaf(
                    image_file=uploaded_file,
                    crop_type=selected_crop,
                    models=models
                )


            # =================================
            # 5. Display Prediction Results
            # =================================

            if not result["is_leaf"]:

                st.error(result["message"])

                st.info(
                    "Please try uploading a clear, well-lit "
                    "image of a leaf."
                )

            elif not result["is_correct_crop"]:

                st.warning(result["message"])

                st.info(
                    f"You selected {crop_type}. "
                    f"Please upload a clear {selected_crop} leaf image."
                )

            else:

                st.success(
                    f"{crop_type} leaf successfully validated!"
                )

                st.subheader("Disease Prediction")

                st.write(
                    f"**Predicted Class:** "
                    f"{result['predicted_class'].title()}"
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

                st.caption(
                    "The confidence score represents the model's "
                    "estimated confidence in its prediction."
                )


        except Exception as e:

            st.error(
                f"An error occurred during prediction: {e}"
            )
