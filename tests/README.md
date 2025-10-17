# Test Suite Documentation

## Panoramica

Questa suite di test fornisce copertura per le classi di processamento documenti del progetto. Include test funzionanti e test avanzati per diversi scenari di utilizzo.

## ⚠️ Stato Attuale dei Test

**Test Funzionanti**: `test_simple.py` - ✅ **TUTTI PASSANO**
- 14 test che verificano funzionalità di base
- Mock appropriati per dipendenze esterne
- Tempo di esecuzione: ~6 secondi
- **Usa questi per validazione rapida**

**Test Avanzati**: `test_document_processors.py`, `test_integration.py`, `test_advanced.py`
- Test più complessi con dipendenze esterne
- Alcuni richiedono aggiornamenti per compatibilità
- Utili per sviluppo futuro

## Struttura dei Test

### 📁 File di Test

- **`test_document_processors.py`** - Test unitari principali per tutte le classi
- **`test_integration.py`** - Test di integrazione end-to-end  
- **`test_advanced.py`** - Test di performance, gestione errori e scenari avanzati

### 🏷️ Marcatori dei Test

- `@pytest.mark.unit` - Test unitari veloci
- `@pytest.mark.integration` - Test di integrazione (possono richiedere risorse esterne)
- `@pytest.mark.slow` - Test che richiedono più tempo (> 5 secondi)
- `@pytest.mark.requires_api_key` - Test che richiedono chiavi API valide

## Esecuzione dei Test

### 🚀 Metodo Rapido (RACCOMANDATO)

```bash
# Esegui test funzionanti (DEFAULT)
python run_tests.py --verbose

# Esegui test semplici con copertura
python run_tests.py --coverage --verbose

# Esegui solo test veloci
python run_tests.py --type fast

# Esegui TUTTI i test (potrebbero fallire)
python run_tests.py --type all
```

## 📝 Test per Sviluppo Quotidiano

```bash
# Test rapidi per verifica funzionalità
python run_tests.py --type simple --verbose

# Test con pytest diretto
python -m pytest tests/test_simple.py -v
```

### 🔧 Metodo Avanzato con pytest

```bash
# Installa dipendenze
pip install pytest pytest-cov pytest-mock

# Tutti i test
pytest

# Solo test unitari
pytest -m unit

# Solo test di integrazione
pytest -m integration

# Test con copertura
pytest --cov=process_docs --cov-report=html

# Test specifico
pytest tests/test_document_processors.py::TestDocumentProcessor::test_init_with_custom_parameters

# Test verbose con output dettagliato
pytest -v -s

# Test paralleli (se hai pytest-xdist)
pytest -n auto
```

## Copertura dei Test

### 📊 Classi Testate

1. **BaseDocumentConverter**
   - ✅ Inizializzazione con parametri validi/invalidi
   - ✅ Conversione documenti (mock e scenari di errore)
   - ✅ Chunking documenti (casi normali e edge cases)
   - ✅ Gestione errori di file non trovati

2. **DocumentProcessor**
   - ✅ Ereditarietà da BaseDocumentConverter
   - ✅ Configurazione LLM e parametri personalizzati
   - ✅ Salvataggio markdown
   - ✅ Split markdown per LLM (gestione token)
   - ✅ Pulizia risposte JSON
   - ✅ Estrazione immagini da markdown
   - ✅ Pipeline completa di processamento
   - ✅ Gestione errori API e JSON malformato

3. **GenericProcessor**
   - ✅ Ereditarietà da BaseDocumentConverter  
   - ✅ Gestione diversi formati di file (.txt, .md, .png, .pdf)
   - ✅ Conversione file a base64
   - ✅ Descrizione immagini con LLM
   - ✅ Conversione PDF a documenti
   - ✅ Pipeline di conversione unificata

### 🎯 Scenari di Test

#### Test Unitari
- Inizializzazione con parametri validi/default
- Gestione errori di input invalidi
- Mock di dipendenze esterne (transformers, OpenAI, Docling)
- Validazione output e metadata
- Edge cases (file vuoti, JSON malformati, etc.)

#### Test di Integrazione  
- Pipeline complete end-to-end
- Processamento file reali (con mock appropriati)
- Batch processing di più documenti
- Compatibilità tra componenti

#### Test di Performance
- ⏱️ Tempo di processamento per documenti grandi
- 💾 Utilizzo memoria con file di grandi dimensioni
- 🧵 Thread safety e processamento concorrente
- 🚀 Scalabilità con carico multiplo

#### Test Avanzati
- 🛡️ Gestione errori di rete (timeout, rate limit)
- 💥 Resilienza a input corrotti
- 🧹 Gestione risorse e cleanup
- 🔧 Configurazione e setup del sistema

## Configurazione Test

### 📋 Variabili d'Ambiente

```bash
# Per test di integrazione con API reali
export OPENAI_API_KEY="your-api-key-here"

# Per test con modelli custom
export HUGGINGFACE_TOKEN="your-hf-token"  # Se necessario
```

### 📦 Dipendenze di Test

```bash
# Dipendenze base
pip install pytest pytest-cov pytest-mock

# Per test con immagini
pip install Pillow

# Per test con PDF
pip install reportlab

# Per test di performance/memoria
pip install psutil

# Per test paralleli (opzionale)
pip install pytest-xdist
```

## Mock e Fixtures

### 🎭 Strategia di Mocking

I test utilizzano mock estensivi per:
- **Transformers/HuggingFace**: Evita download di modelli durante i test
- **OpenAI API**: Evita chiamate API costose e dipendenze di rete
- **Docling**: Mock della conversione documenti per test veloci
- **File I/O**: Controllo completo su operazioni file system

### 🔧 Fixtures Principali

```python
@pytest.fixture
def sample_pdf_path():
    """Path a PDF di esempio per test"""
    
@pytest.fixture  
def temp_directory():
    """Directory temporanea con cleanup automatico"""

@pytest.fixture
def mock_llm_response():
    """Mock response standardizzata da LLM"""
```

## Debugging e Troubleshooting

### 🐛 Problemi Comuni

1. **Test falliti per API Key mancante**
   ```bash
   # Soluzione: imposta la variabile o skippa test che la richiedono
   export OPENAI_API_KEY="test-key"
   pytest -m "not requires_api_key"
   ```

2. **Timeout su test lenti**
   ```bash
   # Soluzione: aumenta timeout o skippa test lenti
   pytest -m "not slow" --timeout=30
   ```

3. **Errori di importazione dipendenze**
   ```bash
   # Soluzione: installa dipendenze mancanti
   pip install -r requirements-test.txt
   ```

### 📊 Report di Copertura

```bash
# Genera report HTML dettagliato
pytest --cov=process_docs --cov-report=html
open htmlcov/index.html

# Report nel terminale
pytest --cov=process_docs --cov-report=term-missing

# Report XML per CI/CD
pytest --cov=process_docs --cov-report=xml
```

## Continuous Integration

### 🔄 GitHub Actions

I test sono configurati per eseguire automaticamente su:
- Push al branch main
- Pull request
- Release tags

Configurazione disponibile in `.github/workflows/test.yml`

### 📈 Metriche di Qualità

- **Copertura minima**: 80%
- **Test success rate**: 100%
- **Performance regression**: < 20% slowdown
- **Memory usage**: < 100MB per test

## Estendere i Test

### ➕ Aggiungere Nuovi Test

1. **Test Unitari**: Aggiungi a `test_document_processors.py`
2. **Test Integrazione**: Aggiungi a `test_integration.py`  
3. **Test Performance**: Aggiungi a `test_advanced.py`

### 🏷️ Convenzioni

```python
class TestNuovaFunzionalita:
    """Test per la nuova funzionalità X."""
    
    def setup_method(self):
        """Setup per ogni test."""
        pass
    
    def test_caso_normale(self):
        """Test caso d'uso normale."""
        pass
    
    def test_edge_case(self):
        """Test caso limite."""
        pass
    
    @pytest.mark.slow
    def test_performance(self):
        """Test di performance (marcato come lento)."""
        pass
```

### 🔍 Best Practices

1. **Nomi descrittivi**: `test_should_return_error_when_file_not_found`
2. **Un assert per test**: Focus su un comportamento specifico
3. **Setup/Teardown**: Usa fixtures per configurazione consistente
4. **Mock estensivo**: Isola le unità sotto test
5. **Documentazione**: Docstring che spiega lo scopo del test

## Supporto

Per domande sui test o per segnalare problemi:
1. Controlla la documentazione sopra
2. Esegui `python run_tests.py --help` per opzioni
3. Verifica la configurazione in `pytest.ini`
4. Consulta i log dettagliati con `pytest -v -s`