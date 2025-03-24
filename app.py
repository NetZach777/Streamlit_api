import os
import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide", page_title="Nessem_Projet")

def icon(emoji: str):
    """Affiche un emoji comme icône de page de style Notion."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🐉")

st.subheader("Beta test Kingdom_IA")

# Utilisation de la clé API à partir des variables d'environnement
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validation de la clé API avant d'initialiser le client
if GROQ_API_KEY is None:
    st.error("Clé API Groq manquante. Vérifiez vos variables d'environnement.")
else:
    client = Groq(api_key=GROQ_API_KEY)

    # Initialiser l'historique des messages et le modèle sélectionné
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Définir les détails des modèles avec les données fournies
    models = {
        "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google"},
        "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B Versatile", "tokens": 32768, "developer": "Meta"},
        "llama-3.1-8b-instant": {"name": "Llama 3.1 8B Instant", "tokens": 8192, "developer": "Meta"},
        "llama-guard-3-8b": {"name": "Llama Guard 3 8B", "tokens": 8192, "developer": "Meta"},
        "llama3-70b-8192": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral"},
        "whisper-large-v3": {"name": "Whisper Large V3", "tokens": 25000, "developer": "OpenAI"},
        "whisper-large-v3-turbo": {"name": "Whisper Large V3 Turbo", "tokens": 25000, "developer": "OpenAI"},
        "qwen-qwq-32b": {"name": "Qwen-QWQ 32B", "tokens": 128000, "developer": "Alibaba Cloud"},
        "mistral-saba-24b": {"name": "Mistral Saba 24B", "tokens": 32000, "developer": "Mistral"},
        "qwen-2.5-coder-32b": {"name": "Qwen 2.5 Coder 32B", "tokens": 128000, "developer": "Alibaba Cloud"},
        "qwen-2.5-32b": {"name": "Qwen 2.5 32B", "tokens": 128000, "developer": "Alibaba Cloud"},
        "deepseek-r1-distill-qwen-32b": {"name": "DeepSeek Distill Qwen 32B", "tokens": 16384, "developer": "DeepSeek"},
        "deepseek-r1-distill-llama-70b-specdec": {"name": "DeepSeek Distill Llama 70B SpecDec", "tokens": 16384, "developer": "DeepSeek"},
        "deepseek-r1-distill-llama-70b": {"name": "DeepSeek Distill Llama 70B", "tokens": 128000, "developer": "DeepSeek"},
        "llama-3.3-70b-specdec": {"name": "Llama 3.3 70B SpecDec", "tokens": 8192, "developer": "Meta"},
        "llama-3.2-1b-preview": {"name": "Llama 3.2 1B Preview", "tokens": 8192, "developer": "Meta"},
        "llama-3.2-3b-preview": {"name": "Llama 3.2 3B Preview", "tokens": 8192, "developer": "Meta"},
        "llama-3.2-11b-vision-preview": {"name": "Llama 3.2 11B Vision Preview", "tokens": 8192, "developer": "Meta"},
        "llama-3.2-90b-vision-preview": {"name": "Llama 3.2 90B Vision Preview", "tokens": 8192, "developer": "Meta"}
    }

    # Disposition pour la sélection du modèle et le curseur max_tokens
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle :",
            options=list(models.keys()),
            format_func=lambda x: f"{models[x]['name']} - {models[x]['tokens']} tokens ({models[x]['developer']})",
            index=0  # Défaut à Distil-Whisper English
        )

    # Détecter le changement de modèle et vider l'historique des messages si le modèle a changé
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col2:
        # Ajuster le curseur max_tokens dynamiquement en fonction du modèle sélectionné
        max_tokens = st.slider(
            "Max Tokens :",
            min_value=512,  # Valeur minimum pour permettre une certaine flexibilité
            max_value=max_tokens_range,
            value=min(8000, max_tokens_range),  # Valeur par défaut ou maximum autorisé si moins
            step=512,
            help=f"Ajustez le nombre de tokens pour la réponse du modèle. Max pour {models[model_option]['name']} : {max_tokens_range}"
        )

    # Affichage des informations supplémentaires sur le modèle
    st.write(f"Modèle sélectionné : **{models[model_option]['name']}**")
    st.write(f"Développeur : {models[model_option]['developer']}")
    st.write(f"Nombre maximal de tokens : {models[model_option]['tokens']}")

    # Avertissement pour les modèles en prévisualisation
    preview_models = [
        "qwen-qwq-32b", "mistral-saba-24b", "qwen-2.5-coder-32b", 
        "qwen-2.5-32b", "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b-specdec",
        "deepseek-r1-distill-llama-70b", "llama-3.3-70b-specdec", 
        "llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-11b-vision-preview", 
        "llama-3.2-90b-vision-preview"
    ]
    if model_option in preview_models:
        st.warning("Ce modèle est en prévisualisation et peut ne pas être stable.")

    # Option pour effacer l'historique des messages
    if st.button("Effacer l'historique"):
        st.session_state.messages = []

    # Afficher les messages de chat de l'historique lors de la réexécution de l'application
    for message in st.session_state.messages:
        avatar = '🐉' if message["role"] == "assistant" else '👨‍👩‍👧‍👧'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Génère le contenu de la réponse du chat à partir de la réponse de l'API Groq."""
        full_content = ""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        yield full_content

    # Gestion des entrées utilisateur et génération de réponses
    if prompt := st.chat_input("Entrez votre message ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)

        # Obtenir la réponse de l'API Groq
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

                # Utiliser la fonction génératrice avec st.write_stream
                full_response = ""
                with st.chat_message("assistant", avatar="👽"):
                    for chunk in generate_chat_responses(chat_completion):
                        full_response += chunk
                        st.markdown(chunk)
                
                # Ajouter la réponse complète à l'historique des messages
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})

        except client.AuthenticationError:
            st.error("Erreur d'authentification. Vérifiez votre clé API.", icon="🚨")
        except client.APIError as api_err:
            st.error(f"Erreur API : {api_err}", icon="🚨")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}", icon="🚨")
