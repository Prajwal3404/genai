import streamlit as st
from googletrans import Translator

# Initialize Google Translator
translator = Translator()

# Streamlit app title
st.title("Kannada to English Translator")

# Text input for Kannada
kannada_text = st.text_area("Enter Kannada text here:")

# Translate button
if st.button("Translate"):
    # Check if input is provided
    if kannada_text.strip() == "":
        st.warning("Please enter some Kannada text to translate.")
    else:
        # Translate text
        translation = translator.translate(kannada_text, src='kn', dest='en')
        english_text = translation.text
        
        # Display the translated English text
        st.write("### Translated English Text:")
        st.write(english_text)

