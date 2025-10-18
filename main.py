# File: main.py
"""
Main entry point for HR RAG system
"""

import argparse
import sys
from src.utils import log_message
from src.ingest import DataIngestion
from src.query import QueryHandler


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="HR RAG System - Free and Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest data from CSV
  python main.py --mode ingest --source csv
  
  # Ingest from Google Sheets
  python main.py --mode ingest --source gsheet
  
  # Incremental update
  python main.py --mode ingest --source csv --incremental
  
  # Query the system
  python main.py --mode query --question "Who has React experience?"
  
  # Query with more context
  python main.py --mode query --question "Compare candidates for ML engineer role" --top-k 10
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['ingest', 'query'],
        help='Operation mode: ingest data or query system'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['csv', 'gsheet'],
        default='csv',
        help='Data source for ingestion (default: csv)'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to CSV file (default: data/sheets.csv)'
    )
    
    parser.add_argument(
        '--sheet-name',
        type=str,
        help='Google Sheet name (default from utils.py)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (for query mode)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of chunks to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only process new/modified candidates'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'ingest':
            # Data ingestion mode
            log_message("="*80)
            log_message("HR RAG SYSTEM - DATA INGESTION")
            log_message("="*80)
            
            ingestion = DataIngestion(incremental=args.incremental)
            ingestion.ingest(
                source=args.source,
                csv_path=args.csv_path,
                sheet_name=args.sheet_name
            )
        
        elif args.mode == 'query':
            # Query mode
            if not args.question:
                print("Error: --question is required for query mode")
                sys.exit(1)
            
            log_message("="*80)
            log_message("HR RAG SYSTEM - QUERY MODE")
            log_message("="*80)
            
            handler = QueryHandler()
            result = handler.query(args.question, top_k=args.top_k)
            
            # Display result
            print(handler.format_output(result))
    
    except KeyboardInterrupt:
        log_message("\nOperation cancelled by user", "INFO")
        sys.exit(0)
    
    except Exception as e:
        log_message(f"\nFatal error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
