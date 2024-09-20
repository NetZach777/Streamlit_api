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

    # Définir les détails des modèles
    models = {
        "llama-3.1-405b-reasoning": {"name": "Llama 3.1 405B", "tokens": 16000, "developer": "Meta"},
        "llama-3.1-70b-versatile": {"name": "Llama 3.1 70B", "tokens": 8000, "developer": "Meta"},
        "llama-3.1-8b-instant": {"name": "Llama 3.1 8B", "tokens": 8000, "developer": "Meta"},
        "llama3-groq-70b-8192-tool-use-preview": {"name": "Llama 3 Groq 70B Tool Use", "tokens": 8192, "developer": "Groq"},
        "llama3-groq-8b-8192-tool-use-preview": {"name": "Llama 3 Groq 8B Tool Use", "tokens": 8192, "developer": "Groq"},
        "llama3-70b-8192": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral"},
        "gemma-7b-it": {"name": "Gemma 7B", "tokens": 8192, "developer": "Google"},
        "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google"},
        "whisper-large-v3": {"name": "Whisper Large V3", "tokens": 8000, "developer": "OpenAI"},
    }

    # Disposition pour la sélection du modèle et le curseur max_tokens
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle :",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=0  # Défaut à Llama 3.1 405B
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
            help=f"Ajustez le nombre maximum de tokens (mots) pour la réponse du modèle. Max pour le modèle sélectionné : {max_tokens_range}"
        )

    # Affichage des informations supplémentaires sur le modèle
    st.write(f"Modèle sélectionné : **{models[model_option]['name']}**")
    st.write(f"Développeur : {models[model_option]['developer']}")
    st.write(f"Nombre maximal de tokens : {models[model_option]['tokens']}")

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
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
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
            st.error(f"Une erreur s'est produite : {e}", icon="🚨")import os
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

    # Définir les détails des modèles
    models = {
        "llama-3.1-405b-reasoning": {"name": "Llama 3.1 405B", "tokens": 16000, "developer": "Meta"},
        "llama-3.1-70b-versatile": {"name": "Llama 3.1 70B", "tokens": 8000, "developer": "Meta"},
        "llama-3.1-8b-instant": {"name": "Llama 3.1 8B", "tokens": 8000, "developer": "Meta"},
        "llama3-groq-70b-8192-tool-use-preview": {"name": "Llama 3 Groq 70B Tool Use", "tokens": 8192, "developer": "Groq"},
        "llama3-groq-8b-8192-tool-use-preview": {"name": "Llama 3 Groq 8B Tool Use", "tokens": 8192, "developer": "Groq"},
        "llama3-70b-8192": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral"},
        "gemma-7b-it": {"name": "Gemma 7B", "tokens": 8192, "developer": "Google"},
        "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google"},
        "whisper-large-v3": {"name": "Whisper Large V3", "tokens": 8000, "developer": "OpenAI"},
    }

    # Disposition pour la sélection du modèle et le curseur max_tokens
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle :",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=0  # Défaut à Llama 3.1 405B
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
            help=f"Ajustez le nombre maximum de tokens (mots) pour la réponse du modèle. Max pour le modèle sélectionné : {max_tokens_range}"
        )

    # Affichage des informations supplémentaires sur le modèle
    st.write(f"Modèle sélectionné : **{models[model_option]['name']}**")
    st.write(f"Développeur : {models[model_option]['developer']}")
    st.write(f"Nombre maximal de tokens : {models[model_option]['tokens']}")

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
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
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
