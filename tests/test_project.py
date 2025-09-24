import os
import pytest
from unittest.mock import patch, MagicMock
from mini_project_1.project import SciFiExplorer  # adjust import based on your file name


# Test 1: Loading documents
def test_load_documents(tmp_path):
    # Arrange: create a fake txt file in a temp directory
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document.")

    explorer = SciFiExplorer.__new__(SciFiExplorer)  # bypass __init__

    # Act
    documents = explorer._load_documents(str(tmp_path))

    # Assert
    assert len(documents) == 1
    assert "This is a test document." in documents[0].page_content


# Test 2: FAISS vector store creation
@patch("mini_project_1.project.FAISS")
@patch("mini_project_1.project.os.path.exists", return_value=False)
def test_get_faiss_vector_store_creates_new(mock_exists, mock_faiss, tmp_path):
    explorer = SciFiExplorer.__new__(SciFiExplorer)

    documents = ["doc1", "doc2"]
    embedding_model = MagicMock()

    # Mock FAISS.from_documents return
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    # Act
    result = explorer._get_faiss_vector_store(documents, embedding_model, str(tmp_path))

    # Assert
    mock_faiss.from_documents.assert_called_once_with(documents=documents, embedding=embedding_model)
    mock_vector_store.save_local.assert_called_once_with(str(tmp_path))
    assert result == mock_vector_store


# Test 3: Ask method with mocked chain
def test_ask_returns_response():
    explorer = SciFiExplorer.__new__(SciFiExplorer)

    # Mock the chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Mocked answer"
    explorer.chain = mock_chain

    # Act
    response = explorer.ask("Who is the main character?")

    # Assert
    mock_chain.invoke.assert_called_once_with("Who is the main character?")
    assert response == "Mocked answer"
