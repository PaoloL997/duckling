"""
Script semplificato per eseguire i test con opzioni predefinite.
Uso: python run_tests.py [options]
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, specific_test=None):
    """
    Esegue i test con configurazioni predefinite.
    
    Args:
        test_type: Tipo di test da eseguire ('unit', 'integration', 'all')
        verbose: Se True, output verboso
        coverage: Se True, calcola la copertura
        specific_test: Test specifico da eseguire
    """
    
    # Base command
    cmd = ["python", "-m", "pytest"]
    
    # Aggiungi opzioni basate sui parametri
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend(["--cov=../process_docs", "--cov-report=term-missing"])
    
    # Seleziona i test da eseguire
    if specific_test:
        cmd.append(specific_test)
    elif test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "simple":
        cmd.append("test_simple.py")
    else:  # all
        cmd.append(".")
    
    print(f"Eseguendo comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Tutti i test sono passati!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Alcuni test sono falliti (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print("❌ pytest non trovato. Installa con: pip install pytest pytest-cov")
        return False


def main():
    parser = argparse.ArgumentParser(description="Esegue i test del progetto")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast", "simple"],
        default="simple",
        help="Tipo di test da eseguire (simple=test funzionanti, all=tutti i test)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Output verboso"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Calcola la copertura del codice"
    )
    parser.add_argument(
        "--test", "-t",
        type=str,
        help="Test specifico da eseguire (es: tests/test_document_processors.py::TestDocumentProcessor::test_init)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Installa le dipendenze per i test"
    )
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("Installando dipendenze per i test...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-cov", "pytest-mock"
        ])
        return
    
    # Verifica che siamo nella directory corretta
    if not Path("../process_docs").exists():
        print("❌ Esegui questo script dalla cartella tests o dalla root del progetto")
        return False
    
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        specific_test=args.test
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()