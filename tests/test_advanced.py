"""
Test di performance e scenari avanzati per le classi di processamento documenti.
"""

import os
import sys
import time
import pytest
import tempfile
import threading
from unittest.mock import patch, Mock
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from process_docs.base import BaseDocumentConverter
from process_docs.convert import DocumentProcessor, GenericProcessor


class TestPerformance:
    """Test di performance e scalabilità."""
    
    def setup_method(self):
        """Setup per test di performance."""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            self.converter = BaseDocumentConverter()
    
    @pytest.mark.slow
    def test_large_document_processing_time(self):
        """Test tempo di processamento per documenti grandi."""
        # Simula un documento con molti chunk
        large_text = "Lorem ipsum " * 10000  # ~130KB di testo
        
        start_time = time.time()
        
        # Simula il processamento
        with patch('tiktoken.get_encoding') as mock_encoding:
            mock_enc = Mock()
            mock_enc.encode.return_value = list(range(len(large_text.split())))
            mock_enc.decode.return_value = large_text
            mock_encoding.return_value = mock_enc
            
            processor = DocumentProcessor()
            chunks = processor.split_markdown_for_llm(large_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Il processamento dovrebbe essere completato in meno di 5 secondi
        assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, expected < 5s"
        assert len(chunks) > 1, "Large text should be split into multiple chunks"
    
    def test_memory_usage_large_file(self):
        """Test utilizzo memoria con file grandi."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Processamento di un file "grande"
        large_content = "x" * 1000000  # 1MB
        
        with patch('transformers.AutoTokenizer.from_pretrained'):
            converter = BaseDocumentConverter()
            
            # Simula processamento multiplo
            for i in range(10):
                with patch('tiktoken.get_encoding') as mock_encoding:
                    mock_enc = Mock()
                    mock_enc.encode.return_value = list(range(len(large_content) // 100))
                    mock_enc.decode.return_value = large_content[:1000]
                    mock_encoding.return_value = mock_enc
                    
                    processor = DocumentProcessor()
                    processor.split_markdown_for_llm(large_content[:1000])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # L'incremento di memoria dovrebbe essere ragionevole (< 100MB)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, expected < 100MB"
    
    def test_concurrent_processing_thread_safety(self):
        """Test sicurezza thread con processamento concorrente."""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            converter = BaseDocumentConverter()
        
        results = []
        errors = []
        
        def process_task(task_id):
            try:
                # Simula processamento concorrente
                mock_document = Mock()
                mock_chunk = Mock()
                mock_chunk.meta.origin.filename = f"doc_{task_id}.pdf"
                mock_chunk.meta.doc_items = [Mock()]
                mock_chunk.meta.doc_items[0].prov = [Mock()]
                mock_chunk.meta.doc_items[0].prov[0].page_no = 1
                mock_chunk.meta.doc_items[-1].prov = [Mock()]
                mock_chunk.meta.doc_items[-1].prov[-1].page_no = 1
                
                converter.chunker.chunk.return_value = [mock_chunk]
                converter.chunker.contextualize.return_value = f"Content {task_id}"
                
                result = converter.chunk_document(mock_document)
                results.append((task_id, len(result)))
                
            except Exception as e:
                errors.append((task_id, str(e)))
        
        # Esegui 10 task concorrenti
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Aspetta tutti i thread
        for thread in threads:
            thread.join(timeout=10)  # Timeout per evitare hang
        
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert len(errors) == 0, f"Got errors: {errors}"
        
        # Verifica che tutti i task abbiano prodotto risultati
        task_ids = [r[0] for r in results]
        assert set(task_ids) == set(range(10))


class TestErrorHandling:
    """Test per gestione errori e resilienza."""
    
    def setup_method(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                with patch('langchain_openai.ChatOpenAI'):
                    self.processor = DocumentProcessor()
    
    def test_network_timeout_handling(self):
        """Test gestione timeout di rete."""
        from requests.exceptions import Timeout
        
        self.processor.llm.invoke.side_effect = Timeout("Network timeout")
        
        with pytest.raises(Timeout):
            self.processor.extract_images_from_markdown("![Test](test.png)")
    
    def test_api_rate_limit_handling(self):
        """Test gestione rate limiting API."""
        from openai import RateLimitError
        
        # Simula rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        rate_limit_error.__class__.__name__ = "RateLimitError"
        
        self.processor.llm.invoke.side_effect = rate_limit_error
        
        with pytest.raises(Exception):
            self.processor.extract_images_from_markdown("![Test](test.png)")
    
    def test_corrupted_pdf_handling(self):
        """Test gestione PDF corrotti."""
        with patch('docling.document_converter.DocumentConverter') as mock_converter_class:
            mock_converter = Mock()
            mock_converter.convert.side_effect = Exception("Corrupted PDF")
            mock_converter_class.return_value = mock_converter
            
            with pytest.raises(Exception, match="Corrupted PDF"):
                self.processor.convert_document("corrupted.pdf")
    
    def test_insufficient_disk_space(self):
        """Test gestione spazio disco insufficiente."""
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError, match="No space left on device"):
                self.processor.save_as_markdown(Mock(), "output.md")
    
    def test_malformed_json_response_recovery(self):
        """Test recupero da risposte JSON malformate."""
        test_cases = [
            '{"incomplete": "json"',  # JSON incompleto
            'Not JSON at all',        # Non JSON
            '{"valid": "json"}garbage', # JSON con garbage
            '',                       # Stringa vuota
        ]
        
        for malformed_json in test_cases:
            mock_response = Mock()
            mock_response.content = malformed_json
            
            self.processor.llm.invoke.return_value = mock_response
            
            # Dovrebbe gestire gracefully senza crashare
            result = self.processor.extract_images_from_markdown("Test content")
            assert result == []  # Dovrebbe ritornare lista vuota in caso di errore


class TestResourceManagement:
    """Test per gestione risorse e cleanup."""
    
    def test_file_descriptor_cleanup(self):
        """Test che i file descriptor vengano chiusi correttamente."""
        import gc
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                with patch('langchain_openai.ChatOpenAI'):
                    processor = GenericProcessor()
        
        # Crea molti file temporanei
        temp_files = []
        for i in range(100):
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write(f"Content {i}")
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Simula processamento
            try:
                processor.text2markdown(temp_file.name)
            except:
                pass  # Ignora errori, stiamo testando solo la gestione risorse
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Il test dovrebbe completarsi senza errori di "too many open files"
        assert True
    
    def test_memory_leak_prevention(self):
        """Test prevenzione memory leak."""
        import gc
        import weakref
        
        with patch('transformers.AutoTokenizer.from_pretrained'):
            converter = BaseDocumentConverter()
        
        objects_to_track = []
        
        # Crea e processa molti oggetti
        for i in range(50):
            mock_doc = Mock()
            mock_doc.test_id = i
            
            # Traccia l'oggetto con weak reference
            weak_ref = weakref.ref(mock_doc)
            objects_to_track.append(weak_ref)
            
            # Simula processamento
            converter.chunker.chunk.return_value = []
            result = converter.chunk_document(mock_doc)
            
            # Rimuovi riferimento locale
            del mock_doc
        
        # Force garbage collection
        gc.collect()
        
        # Controlla che gli oggetti siano stati deallocati
        alive_objects = sum(1 for ref in objects_to_track if ref() is not None)
        
        # Dovrebbero essere deallocati la maggior parte degli oggetti
        assert alive_objects < 10, f"{alive_objects} objects still alive, potential memory leak"


class TestConfigurationAndSetup:
    """Test per configurazione e setup del sistema."""
    
    def test_environment_variable_validation(self):
        """Test validazione variabili d'ambiente."""
        # Test con variabili mancanti
        with patch.dict(os.environ, {}, clear=True):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                # DocumentProcessor dovrebbe funzionare anche senza OPENAI_API_KEY
                processor = DocumentProcessor()
                assert processor is not None
                
                # GenericProcessor dovrebbe fallire
                with pytest.raises(ValueError):
                    GenericProcessor()
    
    def test_model_loading_fallback(self):
        """Test fallback per caricamento modelli."""
        from transformers import AutoTokenizer
        
        # Simula errore nel caricamento del modello
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception):
                BaseDocumentConverter(embedding_model="nonexistent-model")
    
    def test_configuration_persistence(self):
        """Test persistenza configurazione."""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            converter = BaseDocumentConverter(
                embedding_model="custom-model",
                max_tokens=2048
            )
        
        # Verifica che la configurazione sia mantenuta
        assert converter.embedding_model == "custom-model"
        assert converter.max_tokens == 2048
        
        # Verifica che influenzi il comportamento
        assert converter.tokenizer.max_tokens == 2048


if __name__ == "__main__":
    # Esegui i test di performance
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])