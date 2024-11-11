import streamlit as st
import tensorflow as tf
import numpy as np

# Load your pre-trained encoder and decoder models
# Assuming the encoder and decoder models are saved as `.h5` or TensorFlow SavedModel
@st.cache_resource
def load_models():
    # Load encoder and decoder models from the saved paths
    encoder = tf.keras.models.load_model('encoder_model_path')  # Update with actual path
    decoder = tf.keras.models.load_model('decoder_model_path')  # Update with actual path
    return encoder, decoder

encoder_model, decoder_model = load_models()

# Define translation function using encoder-decoder models
def translate_text(input_text, encoder, decoder, tokenizer):
    # Preprocess the input text (tokenization, padding)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='post')
    
    # Pass the input sequence through the encoder
    encoder_output = encoder.predict(input_seq)
    
    # Generate translation using the decoder (apply teacher forcing or greedy decoding as per model)
    translated_output = decoder.predict(encoder_output)

    # Post-process translated output (convert tokens back to text)
    translated_text = " ".join([str(token) for token in translated_output[0]])
    return translated_text

# Streamlit UI
st.title("Kannada to English Machine Translation")

# File uploader for Kannada text and English text files
st.header("Upload Kannada and English Text Files")
kannada_file = st.file_uploader("Upload Kannada Text File", type="txt")
english_file = st.file_uploader("Upload English Text File", type="txt")

# Read files if uploaded
if kannada_file is not None:
    kannada_text = kannada_file.read().decode("utf-8")
    st.text_area("Kannada Text:", kannada_text, height=150)

if english_file is not None:
    english_text = english_file.read().decode("utf-8")
    st.text_area("Reference English Text:", english_text, height=150)

# Text input for interactive translation
st.header("Translate Kannada Text")
input_text = st.text_area("Enter Kannada text here to translate:", "")

# Load tokenizer for preprocessing
# Ensure that you have a suitable tokenizer (you can use a pre-trained tokenizer for the language)
# Here, we're assuming a simple tokenizer for illustration purposes
tokenizer = tf.keras.preprocessing.text.Tokenizer()

if st.button("Translate"):
    if input_text:
        # Translate the input text using the encoder-decoder models
        translated_text = translate_text(input_text, encoder_model, decoder_model, tokenizer)
        st.write("Translated English Text:")
        st.write(translated_text)
    else:
        st.write("Please enter text in Kannada to translate.")

# Optionally, display additional analysis or results from the models
# You could add evaluation metrics, graphs, or intermediate outputs
