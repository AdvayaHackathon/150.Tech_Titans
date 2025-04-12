import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os


        """
                    st.write("**Upload medical images (optional, e.g., X-rays)**")
                    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

                    if uploaded_files:
                                image_paths = save_uploaded_images(uploaded_files, "./test_imgs")
                                st.session_state.image_paths = image_paths
                    submit_button = st.form_submit_button(label="Submit Answers")

                    if submit_button:

                        save_uploaded_images(uploaded_files, output_dir=".//test_imgs")
        """