"""
Test di integrazione end-to-end per le classi di processamento documenti.
Questi test richiedono file reali e possono necessitare di API keys.
"""

import os
import sys
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from process_docs.base import BaseDocumentConverter
from process_docs.convert import DocumentProcessor, GenericProcessor


@pytest.mark.integration
class TestDocumentProcessorIntegration:
    """Test di integrazione per DocumentProcessor."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Setup per test di integrazione."""
        # Skippa se non c'è la chiave API
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping integration tests")
    
    def create_sample_pdf(self, content="Sample PDF Content"):
        """Crea un PDF di esempio per i test."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
        except ImportError:
            pytest.skip("reportlab not installed - cannot create test PDFs")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            c = canvas.Canvas(temp_file.name, pagesize=letter)
            c.drawString(100, 750, content)
            c.showPage()
            c.save()
            return temp_file.name
    
    def test_full_pdf_processing_pipeline(self):
        """Test completo della pipeline di processamento PDF."""
        # Crea un PDF di test
        pdf_path = self.create_sample_pdf("Test document with some content for processing.")
        
        try:
            # Processa con mock per evitare chiamate API reali durante CI
            with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                mock_llm = mock_llm_class.return_value
                mock_response = type('MockResponse', (), {
                    'content': '[{"path": "test.png", "name": "Test Figure", "description": "A test image"}]'
                })()
                mock_llm.invoke.return_value = mock_response
                
                processor = DocumentProcessor()
                
                # Mock del save_as_markdown per evitare problemi con Docling
                with patch.object(processor, 'save_as_markdown', return_value="# Test\n![Test](test.png)"):
                    # Mock della conversione documento
                    mock_doc = type('MockDoc', (), {})()
                    with patch.object(processor, 'convert_document', return_value=mock_doc):
                        # Mock del chunking
                        from langchain.schema import Document
                        mock_chunks = [
                            Document(
                                page_content="Test content chunk",
                                metadata={"path": pdf_path, "pages": [1, 1], "type": "text", "name": "test"}
                            )
                        ]
                        with patch.object(processor, 'chunk_document', return_value=mock_chunks):
                            
                            result = processor.process(pdf_path, "test_output.md")
                            
                            # Verifica i risultati
                            assert len(result) == 2  # 1 text chunk + 1 image
                            
                            text_docs = [doc for doc in result if doc.metadata.get("type") == "text"]
                            image_docs = [doc for doc in result if doc.metadata.get("type") == "image"]
                            
                            assert len(text_docs) == 1
                            assert len(image_docs) == 1
                            
                            assert text_docs[0].page_content == "Test content chunk"
                            assert image_docs[0].metadata["name"] == "Test Figure"
        
        finally:
            # Cleanup
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


@pytest.mark.integration
class TestGenericProcessorIntegration:
    """Test di integrazione per GenericProcessor."""
    
    def setup_method(self):
        """Setup per ogni test."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping integration tests")
    
    def test_text_file_processing(self):
        """Test processamento file di testo."""
        test_content = """# Test Document
        
This is a test document with multiple lines.
It contains some structured content.

## Section 1
Content for section 1.

## Section 2  
Content for section 2.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Mock delle chiamate esterne
                    with patch('langchain_openai.ChatOpenAI'):
                        processor = GenericProcessor()
                        
                        # Mock della load_document per evitare Docling
                        from langchain.schema import Document
                        mock_docs = [
                            Document(
                                page_content="Processed text content",
                                metadata={"source": temp_file.name, "type": "text"}
                            )
                        ]
                        with patch.object(processor, 'load_document', return_value=mock_docs):
                            
                            result = processor.convert(temp_file.name, temp_dir)
                            
                            assert len(result) == 1
                            assert result[0].page_content == "Processed text content"
                            assert result[0].metadata["source"] == temp_file.name
            
            finally:
                os.unlink(temp_file.name)
    
    def test_markdown_file_processing(self):
        """Test processamento file markdown."""
        markdown_content = """# Test Markdown
        
![Test Image](test.png)

Some content with an embedded image.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write(markdown_content)
            temp_file.flush()
            
            try:
                # Mock delle dipendenze esterne
                with patch('langchain_openai.ChatOpenAI'):
                    processor = GenericProcessor()
                    
                    from langchain.schema import Document
                    mock_docs = [
                        Document(
                            page_content=markdown_content,
                            metadata={"source": temp_file.name, "type": "markdown"}
                        )
                    ]
                    with patch.object(processor, 'load_document', return_value=mock_docs):
                        
                        result = processor.convert(temp_file.name)
                        
                        assert len(result) == 1
                        assert "![Test Image](test.png)" in result[0].page_content
            
            finally:
                os.unlink(temp_file.name)
    
    def test_image_file_processing(self):
        """Test processamento file immagine."""
        # Crea un'immagine semplice
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not installed - cannot create test images")
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Crea un'immagine 1x1 pixel
            img = Image.new('RGB', (1, 1), color='red')
            img.save(temp_file.name)
            
            try:
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = mock_llm_class.return_value
                    mock_response = type('MockResponse', (), {'content': 'A red pixel image'})()
                    mock_llm.invoke.return_value = mock_response
                    
                    processor = GenericProcessor()
                    result = processor.convert(temp_file.name)
                    
                    assert len(result) == 1
                    assert result[0].page_content == "A red pixel image"
                    assert result[0].metadata["source"] == temp_file.name
            
            finally:
                os.unlink(temp_file.name)


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestRealAPIIntegration:
    """Test con API reali (richiedono chiavi API valide)."""
    
    def setup_method(self):
        """Setup per test con API reali."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # Verifica che la chiave sia valida (basic check)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key.startswith("sk-"):
            pytest.skip("Invalid OPENAI_API_KEY format")
    
    @pytest.mark.slow
    def test_real_image_description(self):
        """Test descrizione immagine con API reale (SLOW)."""
        # Questo test usa API reali e può essere costoso
        # Usa solo per test manuali o CI con budget API
        
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            pytest.skip("PIL not installed")
        
        # Crea un'immagine con contenuto riconoscibile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            img = Image.new('RGB', (100, 100), color='white')
            draw = ImageDraw.Draw(img)
            draw.rectangle([25, 25, 75, 75], fill='blue', outline='black')
            img.save(temp_file.name)
            
            try:
                processor = GenericProcessor()
                result = processor.image2document(temp_file.name)
                
                # Verifica che abbiamo ottenuto una descrizione
                assert isinstance(result.page_content, str)
                assert len(result.page_content) > 10  # Dovrebbe essere una descrizione sostanziale
                assert result.metadata["source"] == temp_file.name
                
                # La descrizione dovrebbe menzionare forme o colori
                description_lower = result.page_content.lower()
                shape_keywords = ["square", "rectangle", "blue", "shape", "geometric"]
                assert any(keyword in description_lower for keyword in shape_keywords)
            
            finally:
                os.unlink(temp_file.name)


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test di scenari end-to-end completi."""
    
    def test_batch_document_processing(self):
        """Test processamento batch di più documenti."""
        # Crea una directory con diversi tipi di file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Crea file di test
            (temp_path / "doc1.txt").write_text("Content of document 1", encoding='utf-8')
            (temp_path / "doc2.md").write_text("# Document 2\nMarkdown content", encoding='utf-8')
            
            # Crea un'immagine semplice se possibile
            try:
                from PIL import Image
                img = Image.new('RGB', (10, 10), 'red')
                img.save(temp_path / "image1.png")
                image_created = True
            except ImportError:
                image_created = False
            
            # Processa tutti i file
            with patch('langchain_openai.ChatOpenAI'):
                processor = GenericProcessor()
                results = {}
                
                for file_path in temp_path.iterdir():
                    if file_path.is_file():
                        try:
                            # Mock appropriato basato sul tipo di file
                            from langchain.schema import Document
                            if file_path.suffix in ['.txt', '.md']:
                                mock_docs = [Document(
                                    page_content=f"Processed {file_path.name}",
                                    metadata={"source": str(file_path)}
                                )]
                                with patch.object(processor, 'load_document', return_value=mock_docs):
                                    result = processor.convert(str(file_path))
                            elif file_path.suffix in ['.png', '.jpg', '.jpeg']:
                                mock_doc = Document(
                                    page_content=f"Image description for {file_path.name}",
                                    metadata={"source": str(file_path)}
                                )
                                with patch.object(processor, 'image2document', return_value=mock_doc):
                                    result = processor.convert(str(file_path))
                            else:
                                continue
                            
                            results[file_path.name] = result
                        except Exception as e:
                            pytest.fail(f"Failed to process {file_path}: {e}")
                
                # Verifica risultati
                assert len(results) >= 2  # Almeno txt e md
                if image_created:
                    assert len(results) >= 3  # Include anche png
                
                for filename, docs in results.items():
                    assert len(docs) >= 1
                    assert all(isinstance(doc.page_content, str) for doc in docs)


if __name__ == "__main__":
    # Esegui solo test di integrazione
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])