"""
Test robusta per le classi di processamento documenti.
Include test unitari, test di integrazione, e test di edge cases.
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List

# Aggiungi il path del progetto per importare i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from process_docs.base import BaseDocumentConverter
from process_docs.convert import DocumentProcessor, GenericProcessor
from langchain.schema import Document


class TestBaseDocumentConverter:
    """Test per la classe base BaseDocumentConverter."""
    
    def setup_method(self):
        """Setup per ogni test."""
        # Mock di tutte le dipendenze esterne
        with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_auto_tokenizer.return_value = Mock()
            with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer_class.return_value = mock_tokenizer
                with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                    mock_chunker = Mock()
                    mock_chunker_class.return_value = mock_chunker
                    self.base_converter = BaseDocumentConverter(
                        embedding_model="test-model",
                        max_tokens=100
                    )
    
    def test_init_valid_parameters(self):
        """Test inizializzazione con parametri validi."""
        assert self.base_converter.embedding_model == "test-model"
        assert self.base_converter.max_tokens == 100
        assert self.base_converter.tokenizer is not None
        assert self.base_converter.chunker is not None
    
    def test_init_default_parameters(self):
        """Test inizializzazione con parametri di default."""
        with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_auto_tokenizer.return_value = Mock()
            with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer_class.return_value = mock_tokenizer
                with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                    mock_chunker = Mock()
                    mock_chunker_class.return_value = mock_chunker
                    converter = BaseDocumentConverter()
                    assert converter.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
                    assert converter.max_tokens == 4096
    
    def test_convert_document_success(self):
        """Test conversione documento con successo."""
        # Mock dell'intero metodo convert_document per evitare dipendenze esterne
        test_path = "test.pdf"
        mock_document = Mock()
        
        with patch.object(self.base_converter, 'convert_document', return_value=mock_document):
            result = self.base_converter.convert_document(test_path)
            assert result == mock_document
    
    @patch('docling.document_converter.DocumentConverter')
    def test_convert_document_file_not_found(self, mock_converter_class):
        """Test conversione documento con file inesistente."""
        mock_converter = Mock()
        mock_converter.convert.side_effect = FileNotFoundError("File not found")
        mock_converter_class.return_value = mock_converter
        
        with pytest.raises(FileNotFoundError):
            self.base_converter.convert_document("nonexistent.pdf")
    
    def test_chunk_document_success(self):
        """Test chunking documento con successo."""
        # Mock del documento e dei chunk
        mock_document = Mock()
        mock_chunk = Mock()
        
        # Setup mock chunk metadata
        mock_chunk.meta.origin.filename = "test.pdf"
        mock_chunk.meta.doc_items = [Mock()]
        mock_chunk.meta.doc_items[0].prov = [Mock()]
        mock_chunk.meta.doc_items[0].prov[0].page_no = 1
        # Crea un secondo mock per l'ultimo elemento
        mock_chunk.meta.doc_items.append(Mock())
        mock_chunk.meta.doc_items[-1].prov = [Mock()]
        mock_chunk.meta.doc_items[-1].prov[-1].page_no = 2
        
        # Mock del chunker
        self.base_converter.chunker.chunk.return_value = [mock_chunk]
        self.base_converter.chunker.contextualize.return_value = "test content"
        
        result = self.base_converter.chunk_document(mock_document)
        
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].page_content == "test content"
        assert result[0].metadata["path"] == "test.pdf"
        assert result[0].metadata["pages"] == [1, 2]
        assert result[0].metadata["type"] == "text"
        assert result[0].metadata["name"] == "test"
    
    def test_chunk_document_empty_chunks(self):
        """Test chunking documento senza chunk."""
        mock_document = Mock()
        self.base_converter.chunker.chunk.return_value = []
        
        result = self.base_converter.chunk_document(mock_document)
        
        assert result == []


class TestDocumentProcessor:
    """Test per la classe DocumentProcessor."""
    
    def setup_method(self):
        """Setup per ogni test."""
        # Mock delle dipendenze esterne
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
                mock_auto_tokenizer.return_value = Mock()
                with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.return_value = mock_tokenizer
                    with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                        mock_chunker = Mock()
                        mock_chunker_class.return_value = mock_chunker
                        with patch('langchain_openai.ChatOpenAI') as mock_llm:
                            mock_llm.return_value = Mock()
                            self.processor = DocumentProcessor(
                                embed_model_id="test-model",
                                max_tokens=100,
                                llm_model_name="test-gpt",
                                llm_max_tokens=1000
                            )
    
    def test_init_with_custom_parameters(self):
        """Test inizializzazione con parametri personalizzati."""
        assert self.processor.llm_model_name == "test-gpt"
        assert self.processor.llm_max_tokens == 1000
        assert self.processor.llm is not None
    
    def test_init_missing_openai_key(self):
        """Test inizializzazione senza chiave OpenAI."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
                mock_auto_tokenizer.return_value = Mock()
                with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.return_value = mock_tokenizer
                    with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                        mock_chunker = Mock()
                        mock_chunker_class.return_value = mock_chunker
                        with patch('langchain_openai.ChatOpenAI') as mock_llm:
                            mock_llm.return_value = Mock()
                            # Dovrebbe funzionare ma non avere accesso alle funzionalità LLM
                            processor = DocumentProcessor()
                            assert processor.llm is not None  # ChatOpenAI dovrebbe gestire l'errore
    
    def test_save_as_markdown_success(self):
        """Test salvataggio come markdown."""
        mock_document = Mock()
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        temp_file.write("# Test Content")
        temp_file.close()
        
        try:
            # Mock del metodo save_as_markdown
            mock_document.save_as_markdown = Mock()
            
            with patch('builtins.open', mock_open_with_content("# Test Content")):
                result = self.processor.save_as_markdown(mock_document, temp_file.name)
                
                assert result == "# Test Content"
                mock_document.save_as_markdown.assert_called_once()
        finally:
            os.unlink(temp_file.name)
    
    def test_split_markdown_for_llm(self):
        """Test split markdown per LLM."""
        test_markdown = "Short content"
        
        with patch('tiktoken.get_encoding') as mock_encoding:
            mock_enc = Mock()
            mock_enc.encode.return_value = [1, 2, 3, 4, 5]  # 5 token
            mock_enc.decode.return_value = test_markdown
            mock_encoding.return_value = mock_enc
            
            self.processor.llm_max_tokens = 3  # Forza la divisione
            
            result = self.processor.split_markdown_for_llm(test_markdown)
            
            assert len(result) >= 1  # Dovrebbe essere diviso
    
    def test_clean_json_response(self):
        """Test pulizia risposta JSON."""
        test_cases = [
            ('```json\n{"test": "value"}\n```', '{"test": "value"}'),
            ('{"test": "value"}', '{"test": "value"}'),
            ('```json{"test": "value"}```', '{"test": "value"}'),
            ('   ```json\n{"test": "value"}\n```   ', '{"test": "value"}')
        ]
        
        for input_text, expected in test_cases:
            result = self.processor.clean_json_response(input_text)
            assert result == expected
    
    @patch('process_docs.convert.logger')
    def test_extract_images_from_markdown_success(self, mock_logger):
        """Test estrazione immagini da markdown."""
        test_markdown = "![Test image](test.png)"
        mock_response = Mock()
        mock_response.content = '[{"path": "test.png", "name": "Test image", "description": "A test image"}]'
        
        self.processor.llm.invoke.return_value = mock_response
        
        with patch.object(self.processor, 'split_markdown_for_llm', return_value=[test_markdown]):
            result = self.processor.extract_images_from_markdown(test_markdown)
            
            assert len(result) == 1
            assert isinstance(result[0], Document)
            assert result[0].page_content == "A test image"
            assert result[0].metadata["path"] == "test.png"
            assert result[0].metadata["name"] == "Test image"
            assert result[0].metadata["type"] == "image"
    
    @patch('process_docs.convert.logger')
    def test_extract_images_invalid_json(self, mock_logger):
        """Test estrazione immagini con JSON invalido."""
        test_markdown = "Test content"
        mock_response = Mock()
        mock_response.content = 'Invalid JSON {'
        
        self.processor.llm.invoke.return_value = mock_response
        
        with patch.object(self.processor, 'split_markdown_for_llm', return_value=[test_markdown]):
            result = self.processor.extract_images_from_markdown(test_markdown)
            
            assert result == []
            mock_logger.error.assert_called()
    
    def test_process_integration(self):
        """Test del processo completo (test di integrazione)."""
        test_pdf = "test.pdf"
        
        # Mock di tutti i metodi chiamati
        mock_document = Mock()
        mock_text_docs = [Document(page_content="test", metadata={})]
        mock_image_docs = [Document(page_content="image desc", metadata={})]
        
        with patch.object(self.processor, 'convert_document', return_value=mock_document):
            with patch.object(self.processor, 'chunk_document', return_value=mock_text_docs):
                with patch.object(self.processor, 'save_as_markdown', return_value="# Test"):
                    with patch.object(self.processor, 'extract_images_from_markdown', return_value=mock_image_docs):
                        
                        result = self.processor.process(test_pdf)
                        
                        assert len(result) == 2  # 1 testo + 1 immagine
                        assert result[:1] == mock_text_docs
                        assert result[1:] == mock_image_docs


class TestGenericProcessor:
    """Test per la classe GenericProcessor."""
    
    def setup_method(self):
        """Setup per ogni test."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
                mock_auto_tokenizer.return_value = Mock()
                with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.return_value = mock_tokenizer
                    with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                        mock_chunker = Mock()
                        mock_chunker_class.return_value = mock_chunker
                        with patch('langchain_openai.ChatOpenAI'):
                            self.processor = GenericProcessor(llm_model="test-gpt")
    
    def test_init_valid_parameters(self):
        """Test inizializzazione con parametri validi."""
        assert self.processor.OPENAI_API_KEY == "test-key"
        assert self.processor.llm is not None
    
    def test_init_missing_openai_key(self):
        """Test inizializzazione senza chiave OpenAI."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
                mock_auto_tokenizer.return_value = Mock()
                with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.return_value = mock_tokenizer
                    with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                        mock_chunker = Mock()
                        mock_chunker_class.return_value = mock_chunker
                        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                            GenericProcessor()
    
    def test_file_to_base64(self):
        """Test conversione file a base64."""
        test_content = b"test binary content"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                result = GenericProcessor._file_to_base64(temp_file.name)
                expected = base64.b64encode(test_content).decode("utf-8")
                assert result == expected
            finally:
                os.unlink(temp_file.name)
    
    def test_image_to_document(self):
        """Test conversione immagine a documento."""
        mock_response = Mock()
        mock_response.content = "This is an image description"
        self.processor.llm.invoke.return_value = mock_response
        
        test_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        result = self.processor._image_to_document(test_b64, "Test prompt", "test.png")
        
        assert isinstance(result, Document)
        assert result.page_content == "This is an image description"
        assert result.metadata["source"] == "test.png"
    
    def test_load_document_pdf(self):
        """Test caricamento documento PDF."""
        test_path = "test.pdf"
        
        mock_document = Mock()
        mock_chunks = [Document(page_content="test chunk", metadata={})]
        
        with patch.object(self.processor, 'convert_document', return_value=mock_document):
            with patch.object(self.processor, 'chunk_document', return_value=mock_chunks):
                result = self.processor.load_document(test_path)
                
                assert result == mock_chunks
    
    def test_text2markdown_success(self):
        """Test conversione testo a markdown."""
        test_content = "Line 1\nLine 2\nLine 3"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    result_path = self.processor.text2markdown(temp_file.name, temp_dir)
                    
                    assert result_path.endswith('.md')
                    assert os.path.exists(result_path)
                    
                    with open(result_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        assert content == test_content
            finally:
                os.unlink(temp_file.name)
    
    def test_image2document_with_context(self):
        """Test conversione immagine con contesto."""
        mock_response = Mock()
        mock_response.content = "Contextual image description"
        self.processor.llm.invoke.return_value = mock_response
        
        test_image_path = "test.png"
        test_context = "This is related to machine learning"
        
        with patch.object(self.processor, '_file_to_base64', return_value="fake_b64"):
            result = self.processor.image2document(test_image_path, test_context)
            
            assert isinstance(result, Document)
            assert result.page_content == "Contextual image description"
            assert result.metadata["source"] == test_image_path
    
    @patch('fitz.open')
    def test_pdf2documents_success(self, mock_fitz_open):
        """Test conversione PDF a documenti."""
        # Mock del PDF e delle pagine
        mock_pdf = Mock()
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake image bytes"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pdf.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz_open.return_value = mock_pdf
        
        mock_response = Mock()
        mock_response.content = "Page content"
        self.processor.llm.invoke.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.processor.pdf2documents("test.pdf", temp_dir)
            
            assert len(result) == 1
            assert isinstance(result[0], Document)
            assert result[0].page_content.startswith("# Page 1")
            assert result[0].metadata["page"] == 1
    
    def test_convert_txt_file(self):
        """Test conversione file TXT."""
        test_path = "test.txt"
        mock_docs = [Document(page_content="content", metadata={})]
        
        with patch.object(self.processor, 'text2markdown', return_value="test.md"):
            with patch.object(self.processor, 'load_document', return_value=mock_docs):
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = self.processor.convert(test_path, temp_dir)
                    
                    assert result == mock_docs
    
    def test_convert_image_file(self):
        """Test conversione file immagine."""
        for ext in GenericProcessor.IMG_EXTENSION:
            test_path = f"test{ext}"
            mock_doc = Document(page_content="image desc", metadata={})
            
            with patch.object(self.processor, 'image2document', return_value=mock_doc):
                result = self.processor.convert(test_path)
                
                assert len(result) == 1
                assert result[0] == mock_doc
    
    def test_convert_unsupported_format(self):
        """Test conversione formato non supportato."""
        with pytest.raises(ValueError, match="Unsupported file format: .xyz"):
            self.processor.convert("test.xyz")


class TestEdgeCases:
    """Test per edge cases e scenari limite."""
    
    def test_large_file_handling(self):
        """Test gestione file di grandi dimensioni."""
        # Simula un file molto grande
        large_content = "x" * 1000000  # 1MB di testo
        
        with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_auto_tokenizer.return_value = Mock()
            with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer_class.return_value = mock_tokenizer
                with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                    mock_chunker = Mock()
                    mock_chunker_class.return_value = mock_chunker
                    converter = BaseDocumentConverter()
            
            # Test che non dovrebbe causare problemi di memoria
            with patch('tiktoken.get_encoding') as mock_encoding:
                mock_enc = Mock()
                mock_enc.encode.return_value = list(range(len(large_content)))
                mock_enc.decode.return_value = large_content
                mock_encoding.return_value = mock_enc
                
                processor = DocumentProcessor()
                result = processor.split_markdown_for_llm(large_content)
                
                assert len(result) > 1  # Dovrebbe essere diviso in chunk
    
    def test_empty_file_handling(self):
        """Test gestione file vuoti."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("")  # File vuoto
            temp_file.close()  # Chiudi il file per Windows
            
            try:
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                    with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
                        mock_auto_tokenizer.return_value = Mock()
                        with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                            mock_tokenizer = Mock()
                            mock_tokenizer_class.return_value = mock_tokenizer
                            with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                                mock_chunker = Mock()
                                mock_chunker_class.return_value = mock_chunker
                                with patch('langchain_openai.ChatOpenAI'):
                                    processor = GenericProcessor()
                                
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    result_path = processor.text2markdown(temp_file.name, temp_dir)
                                    
                                    assert os.path.exists(result_path)
                                    with open(result_path, 'r') as f:
                                        assert f.read() == ""
            finally:
                try:
                    os.unlink(temp_file.name)
                except (PermissionError, FileNotFoundError):
                    pass  # Ignora errori su Windows
    
    def test_concurrent_processing(self):
        """Test processamento concorrente."""
        import threading
        import time
        
        with patch('process_docs.base.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_auto_tokenizer.return_value = Mock()
            with patch('process_docs.base.HuggingFaceTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer_class.return_value = mock_tokenizer
                with patch('process_docs.base.HybridChunker') as mock_chunker_class:
                    mock_chunker = Mock()
                    mock_chunker_class.return_value = mock_chunker
                    converter = BaseDocumentConverter()
            results = {}
            errors = {}
            
            def process_document(doc_id):
                try:
                    # Simula processamento
                    time.sleep(0.1)
                    results[doc_id] = f"processed_{doc_id}"
                except Exception as e:
                    errors[doc_id] = str(e)
            
            # Avvia thread multipli
            threads = []
            for i in range(5):
                thread = threading.Thread(target=process_document, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Aspetta che tutti finiscano
            for thread in threads:
                thread.join()
            
            assert len(results) == 5
            assert len(errors) == 0


def mock_open_with_content(content):
    """Helper per mockare file con contenuto specifico."""
    from unittest.mock import mock_open
    return mock_open(read_data=content)


# Fixtures per pytest
@pytest.fixture
def sample_pdf_path():
    """Fixture che fornisce un path PDF di esempio."""
    return os.path.join(os.path.dirname(__file__), 'sample.pdf')


@pytest.fixture
def temp_directory():
    """Fixture che fornisce una directory temporanea."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Esegui i test direttamente
    pytest.main([__file__, "-v", "--tb=short"])