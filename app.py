import os
import streamlit as st
from typing import Generator
from groq import Groq
from PIL import Image
import time

st.set_page_config(page_icon="💬", layout="wide", page_title="Nessem_Projet")

def icon(emoji: str):
    """Affiche un emoji comme icône de page de style Notion."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🐉")

st.subheader("Beta test Kingdom_IA")

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if GROQ_API_KEY is None:
    st.error("Clé API Groq manquante. Vérifiez vos variables d'environnement.")
else:
    client = Groq(api_key=GROQ_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    
 # Définir les détails des modèles avec les données fournies
    models = {
    # Modèles de traitement du langage naturel (NLP)
    "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google", "type": "chat"},
    "meta-llama-3.3-70b-versatile": {"name": "Llama 3.3 70B Versatile", "tokens": 32768, "developer": "Meta", "type": "chat"},
    "meta-llama-3.1-8b-instant": {"name": "Llama 3.1 8B Instant", "tokens": 8192, "developer": "Meta", "type": "chat"},
    "meta-llama-guard-3-8b": {"name": "Llama Guard 3 8B", "tokens": 8192, "developer": "Meta", "type": "chat"},
    "meta-llama-3-70b": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta", "type": "chat"},
    "meta-llama-3-8b": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta", "type": "chat"},
    "mistral-8x7b": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral", "type": "chat"},
    "qwen-qwq-32b": {"name": "Qwen-QWQ 32B", "tokens": 128000, "developer": "Alibaba Cloud", "type": "chat"},
    "mistral-saba-24b": {"name": "Mistral Saba 24B", "tokens": 32000, "developer": "Mistral", "type": "chat"},
    "qwen-2.5-coder-32b": {"name": "Qwen 2.5 Coder 32B", "tokens": 128000, "developer": "Alibaba Cloud", "type": "chat"},
    "qwen-2.5-32b": {"name": "Qwen 2.5 32B", "tokens": 128000, "developer": "Alibaba Cloud", "type": "chat"},
    "deepseek-r1-distill-qwen-32b": {"name": "DeepSeek Distill Qwen 32B", "tokens": 16384, "developer": "DeepSeek", "type": "chat"},
    "deepseek-r1-distill-llama-70b": {"name": "DeepSeek Distill Llama 70B", "tokens": 128000, "developer": "DeepSeek", "type": "chat"},
    
    # Modèles de vision
    "meta-llama-3.2-11b-vision-preview": {"name": "Llama 3.2 11B Vision Preview", "tokens": 8192, "developer": "Meta", "type": "vision"},
    "meta-llama-3.2-90b-vision-preview": {"name": "Llama 3.2 90B Vision Preview", "tokens": 8192, "developer": "Meta", "type": "vision"},
    
    # Modèles de reconnaissance vocale
    "openai-whisper-large-v3": {"name": "Whisper Large V3", "tokens": 25000, "developer": "OpenAI", "type": "audio"},
    "openai-whisper-large-v3-turbo": {"name": "Whisper Large V3 Turbo", "tokens": 25000, "developer": "OpenAI", "type": "audio"}
}
    
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle :",
            options=list(models.keys()),
            format_func=lambda x: f"{models[x]['name']} - {models[x]['tokens']} tokens ({models[x]['developer']})",
            index=0
        )

    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col2:
        max_tokens = st.slider(
            "Max Tokens :",
            min_value=512,
            max_value=max_tokens_range,
            value=min(8000, max_tokens_range),
            step=512
        )

    st.write(f"Modèle sélectionné : **{models[model_option]['name']}**")
    st.write(f"Développeur : {models[model_option]['developer']}")
    st.write(f"Nombre maximal de tokens : {models[model_option]['tokens']}")

    if models[model_option]["type"] == "vision":
        uploaded_image = st.file_uploader("Upload une image", type=["png", "jpg", "jpeg"])
    else:
        uploaded_image = None

    if st.button("Effacer l'historique"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = '🐉' if message["role"] == "assistant" else '👤'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        full_content = ""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        yield full_content

    if prompt := st.chat_input("Entrez votre message ici...") or uploaded_image:
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Image téléchargée", use_column_width=True)
            prompt = "Analyse cette image."

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)

        try:
            with st.spinner('Génération de la réponse...'):
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages
                    ],
                    max_tokens=max_tokens,
                    stream=True
                )
                
                full_response = ""
                with st.chat_message("assistant", avatar="👽"):
                    for chunk in generate_chat_responses(chat_completion):
                        full_response += chunk
                        st.markdown(chunk)
                        time.sleep(0.05)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except client.AuthenticationError:
            st.error("Erreur d'authentification. Vérifiez votre clé API.", icon="🚨")
        except client.APIError as api_err:
            st.error(f"Erreur API : {api_err}", icon="🚨")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}", icon="🚨")
