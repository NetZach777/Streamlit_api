import pytest
from unittest.mock import patch, MagicMock
import app  # Importer l'application

# Test de la fonction generate_chat_responses
def test_generate_chat_responses():
    mock_response = MagicMock()
    mock_response.choices[0].delta.content = "Bonjour!"
    
    responses = list(app.generate_chat_responses([mock_response]))
    assert responses == ["Bonjour!"]

# Test de la fonction principale de l'application (vous pouvez ajouter plus de détails ici)
@patch('app.client.chat.completions.create')
def test_chat_completion(mock_create):
    mock_create.return_value = [MagicMock(
        choices=[MagicMock(delta=MagicMock(content="Salut!"))]
    )]
    
    responses = app.generate_chat_responses(mock_create())
    assert list(responses) == ["Salut!"]