import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Available language pairs for translation
LANGUAGE_PAIRS = {
    "English to French": ("en", "fr"),
    "French to English": ("fr", "en"),
    "English to German": ("en", "de"),
    "German to English": ("de", "en"),
    "English to Spanish": ("en", "es"),
    "Spanish to English": ("es", "en"),
}

def load_model(src_lang, tgt_lang):
    """Load the MarianMT model and tokenizer for translation"""
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, src_lang, tgt_lang):
    """Translate the given text from source to target language"""
    tokenizer, model = load_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title("TransLingua: AI-Powered Multi-Language Translator")
st.write("Translate text seamlessly between multiple languages using AI.")

# User input
text_input = st.text_area("Enter text to translate:", height=150)
selected_pair = st.selectbox("Select Language Pair:", list(LANGUAGE_PAIRS.keys()))

if st.button("Translate"):
    if text_input.strip():
        src, tgt = LANGUAGE_PAIRS[selected_pair]
        translation = translate_text(text_input, src, tgt)
        st.success("Translated Text:")
        st.write(translation)
    else:
        st.warning("Please enter text to translate.")

# Run Streamlit app with: streamlit run app.py