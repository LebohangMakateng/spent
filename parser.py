"""
Bank Statement Parser - Spent Money Detective
Handles loading, cleaning, and normalizing bank statement data from CSV and PDF files.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import pdfplumber
from io import BytesIO

class BankStatementParser:
    """Parse and clean bank statement CSV files."""
    
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
        
    def load_csv(self, file_path):
        """Load CSV file and handle common encoding issues."""
        try:
            # Try UTF-8 first
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try UTF-8 with BOM
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                # Fallback to latin-1
                df = pd.read_csv(file_path, encoding='latin-1')
        
        self.raw_data = df
        return df
    
    def clean_and_normalize(self, df):
        """Clean and normalize the bank statement data."""
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove BOM character if present in column names
        cleaned_df.columns = [col.replace('\ufeff', '') for col in cleaned_df.columns]
        
        # Standardize column names (case insensitive matching)
        column_mapping = self._detect_columns(cleaned_df.columns)
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['Date', 'Description', 'Amount', 'Balance']
        for col in required_cols:
            if col not in cleaned_df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Clean the data
        cleaned_df = self._clean_dates(cleaned_df)
        cleaned_df = self._clean_amounts(cleaned_df)
        cleaned_df = self._clean_descriptions(cleaned_df)
        cleaned_df = self._remove_junk_rows(cleaned_df)
        
        # Sort by date
        cleaned_df = cleaned_df.sort_values('Date').reset_index(drop=True)
        
        self.cleaned_data = cleaned_df
        return cleaned_df
    
    def _detect_columns(self, columns):
        """Detect and map column names to standard format."""
        mapping = {}
        
        for col in columns:
            col_lower = col.lower().strip()
            
            if 'date' in col_lower:
                mapping[col] = 'Date'
            elif 'description' in col_lower or 'detail' in col_lower or 'narrative' in col_lower:
                mapping[col] = 'Description'
            elif 'amount' in col_lower and 'balance' not in col_lower:
                mapping[col] = 'Amount'
            elif 'balance' in col_lower:
                mapping[col] = 'Balance'
            elif 'charge' in col_lower or 'fee' in col_lower:
                mapping[col] = 'AccruedBankCharge'
        
        return mapping
    
    def _clean_dates(self, df):
        """Clean and standardize date formats."""
        # Handle various date formats common in bank statements
        date_formats = [
            '%d/%m/%Y',    # 27/05/2023
            '%d-%m-%Y',    # 27-05-2023
            '%Y/%m/%d',    # 2023/05/27
            '%Y-%m-%d',    # 2023-05-27
            '%d %b %Y',    # 27 May 2023
            '%d %B %Y',    # 27 May 2023
            '%d %b',       # 27 May (assume current year)
            '%d %B',       # 27 May (assume current year)
        ]
        
        # Try to parse dates with multiple formats
        def parse_date_flexible(date_str):
            if pd.isna(date_str) or str(date_str).strip() == '':
                return pd.NaT
            
            date_str = str(date_str).strip()
            
            # Try each format
            for fmt in date_formats:
                try:
                    parsed_date = pd.to_datetime(date_str, format=fmt)
                    # If format doesn't include year, assume current year
                    if '%Y' not in fmt:
                        parsed_date = parsed_date.replace(year=pd.Timestamp.now().year)
                    return parsed_date
                except (ValueError, TypeError):
                    continue
            
            # If all formats fail, try pandas' flexible parser
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT
        
        # Apply flexible date parsing
        df['Date'] = df['Date'].apply(parse_date_flexible)
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        return df
    
    def _clean_amounts(self, df):
        """Clean and standardize amount formats."""
        # Convert Amount to numeric, handling various formats
        df['Amount'] = df['Amount'].astype(str)
        # Handle non-breaking spaces and other whitespace
        df['Amount'] = df['Amount'].str.replace('\xa0', ' ', regex=False)
        df['Amount'] = df['Amount'].str.replace(',', '')  # Remove commas
        df['Amount'] = df['Amount'].str.replace(' ', '')   # Remove spaces
        # Handle negative amounts in parentheses
        df['Amount'] = df['Amount'].str.replace('(', '-', regex=False)
        df['Amount'] = df['Amount'].str.replace(')', '', regex=False)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Clean Balance column similarly
        if 'Balance' in df.columns:
            df['Balance'] = df['Balance'].astype(str)
            df['Balance'] = df['Balance'].str.replace('\xa0', ' ', regex=False)
            df['Balance'] = df['Balance'].str.replace(',', '')
            df['Balance'] = df['Balance'].str.replace(' ', '')
            df['Balance'] = df['Balance'].str.replace('(', '-')  # Handle negative balances in parentheses
            df['Balance'] = df['Balance'].str.replace(')', '')
            df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        
        return df
    
    def _clean_descriptions(self, df):
        """Clean and standardize transaction descriptions."""
        df['Description'] = df['Description'].astype(str)
        
        # Remove extra whitespace
        df['Description'] = df['Description'].str.strip()
        
        # Remove empty descriptions
        df = df[df['Description'] != '']
        df = df[df['Description'] != 'nan']
        
        return df
    
    def _remove_junk_rows(self, df):
        """Remove junk rows like headers, footers, empty rows."""
        # Remove rows where Amount is NaN (likely junk)
        df = df.dropna(subset=['Amount'])
        
        # Remove rows where Amount is 0 and Description is empty or generic
        junk_descriptions = ['', 'nan', 'opening balance', 'closing balance']
        mask = ~((df['Amount'] == 0) & (df['Description'].str.lower().isin(junk_descriptions)))
        df = df[mask]
        
        # Remove duplicate transactions (same date, description, amount)
        df = df.drop_duplicates(subset=['Date', 'Description', 'Amount'], keep='first')
        
        return df
    
    def get_transaction_summary(self):
        """Get summary statistics of the parsed transactions."""
        if self.cleaned_data is None:
            return None
        
        df = self.cleaned_data
        
        # Safe date formatting
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'Unknown',
                'end': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'Unknown'
            },
            'total_debits': df[df['Amount'] < 0]['Amount'].sum(),
            'total_credits': df[df['Amount'] > 0]['Amount'].sum(),
            'net_change': df['Amount'].sum(),
            'largest_expense': df[df['Amount'] < 0]['Amount'].min(),
            'largest_income': df[df['Amount'] > 0]['Amount'].max()
        }
        
        return summary

class PDFBankStatementParser:
    """Parse and extract transactions from PDF bank statements."""
    
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
    
    def load_pdf(self, file_path_or_buffer):
        """Load PDF file and extract transaction tables."""
        try:
            if isinstance(file_path_or_buffer, (str, bytes)):
                # It's a file path or bytes
                with pdfplumber.open(file_path_or_buffer) as pdf:
                    return self._extract_transactions_from_pdf(pdf)
            else:
                # It's a file-like object (uploaded file)
                with pdfplumber.open(file_path_or_buffer) as pdf:
                    return self._extract_transactions_from_pdf(pdf)
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
    
    def _extract_transactions_from_pdf(self, pdf):
        """Extract transaction data from all pages of the PDF."""
        all_transactions = []
        debug_info = []
        
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract tables from the page
                tables = page.extract_tables()
                debug_info.append(f"Page {page_num + 1}: Found {len(tables) if tables else 0} tables")
                
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            debug_info.append(f"  Table {table_idx + 1}: {len(table)} rows, {len(table[0]) if table[0] else 0} columns")
                            # Show first few rows for debugging
                            for row_idx, row in enumerate(table[:3]):  # First 3 rows
                                debug_info.append(f"    Row {row_idx}: {row}")
                            
                            # Process each table
                            transactions = self._process_table(table, page_num)
                            if transactions:
                                debug_info.append(f"    -> Extracted {len(transactions)} transactions")
                                all_transactions.extend(transactions)
                            else:
                                # Show why no transactions were extracted
                                header_idx = self._find_header_row(table)
                                if header_idx is not None:
                                    headers = [str(h).strip() if h is not None else '' for h in table[header_idx]]
                                    mapping = self._map_pdf_columns(headers)
                                    debug_info.append(f"    -> Found headers: {headers}")
                                    debug_info.append(f"    -> Column mapping: {mapping}")
                                    debug_info.append(f"    -> No transactions extracted (check data rows)")
                                else:
                                    debug_info.append(f"    -> No transaction header found")
            except Exception as e:
                debug_info.append(f"Warning: Error processing page {page_num + 1}: {str(e)}")
                continue
        
        if not all_transactions:
            # Provide more helpful debugging information
            total_tables = sum(len(page.extract_tables()) for page in pdf.pages if page.extract_tables())
            error_msg = f"No transaction data found in PDF. Found {total_tables} tables across {len(pdf.pages)} pages.\n"
            error_msg += "Debug info:\n" + "\n".join(debug_info[:10])  # Show first 10 debug lines
            error_msg += "\n\nPlease ensure the PDF contains a transaction table with recognizable headers like Date, Description, Amount, Balance."
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)
        self.raw_data = df
        return df
    
    def _process_table(self, table, page_num):
        """Process a table and extract transaction data."""
        transactions = []
        
        # Find header row (contains Date, Description, Amount, Balance)
        header_row_idx = self._find_header_row(table)
        
        if header_row_idx is None:
            return transactions
        
        headers = table[header_row_idx]
        
        # Clean headers and handle None values
        headers = [str(h).strip() if h is not None else '' for h in headers]
        
        # Map headers to standard columns
        column_mapping = self._map_pdf_columns(headers)
        
        if not column_mapping:
            return transactions
        
        # Process data rows
        for row_idx in range(header_row_idx + 1, len(table)):
            row = table[row_idx]
            if not row or len(row) < len(headers):
                continue
            
            # Clean row data
            row_data = [str(cell).strip() if cell else '' for cell in row]
            
            # Skip empty rows or rows with no date
            if not row_data or not any(row_data):
                continue
            
            # Create transaction record
            transaction = {}
            for original_col, standard_col in column_mapping.items():
                col_idx = headers.index(original_col) if original_col in headers else -1
                if col_idx >= 0 and col_idx < len(row_data):
                    value = row_data[col_idx].strip() if row_data[col_idx] else ''
                    if value:  # Only add non-empty values
                        transaction[standard_col] = value
            
            # Only add if we have essential data (Date and either Amount or Balance)
            has_date = transaction.get('Date') and transaction.get('Date').strip()
            has_amount = transaction.get('Amount') and transaction.get('Amount').strip()
            has_balance = transaction.get('Balance') and transaction.get('Balance').strip()
            
            if has_date and (has_amount or has_balance):
                transactions.append(transaction)
        
        return transactions
    
    def _find_header_row(self, table):
        """Find the row containing transaction headers."""
        if not table:
            return None
        
        # Look for common header patterns (more comprehensive)
        header_indicators = [
            'date', 'description', 'amount', 'balance', 'debit', 'credit',
            'transaction', 'detail', 'narrative', 'reference', 'value',
            'withdrawal', 'deposit', 'payment', 'transfer'
        ]
        
        for row_idx, row in enumerate(table):
            if not row:
                continue
            
            row_text = ' '.join([str(cell).lower().strip() if cell else '' for cell in row])
            
            # Count how many header indicators we find
            matches = sum(1 for indicator in header_indicators if indicator in row_text)
            
            # If we find at least 2 indicators (lowered threshold), it's likely a header row
            if matches >= 2:
                return row_idx
        
        return None
    
    def _map_pdf_columns(self, headers):
        """Map PDF column headers to standard format."""
        mapping = {}
        
        for i, header in enumerate(headers):
            if not header or header.strip() == '':
                continue
            
            header_lower = header.lower().strip()
            
            # Date column
            if 'date' in header_lower:
                mapping[header] = 'Date'
            # Description column
            elif any(word in header_lower for word in ['description', 'detail', 'narrative', 'transaction']):
                mapping[header] = 'Description'
            # Amount column (could be combined or separate debit/credit)
            elif 'amount' in header_lower:
                mapping[header] = 'Amount'
            # Balance column
            elif 'balance' in header_lower:
                mapping[header] = 'Balance'
            # Separate debit/credit columns
            elif 'debit' in header_lower:
                mapping[header] = 'Debit'
            elif 'credit' in header_lower:
                mapping[header] = 'Credit'
            # Bank charges or fees column
            elif any(word in header_lower for word in ['charge', 'fee', 'accrued']):
                mapping[header] = 'AccruedBankCharge'
        
        # Ensure we have at least Date and Amount/Debit/Credit
        has_date = 'Date' in mapping.values()
        has_amount = any(col in mapping.values() for col in ['Amount', 'Debit', 'Credit'])
        has_balance = 'Balance' in mapping.values()
        
        # We need at least Date and either Amount or Balance to proceed
        if not (has_date and (has_amount or has_balance)):
            return {}
        
        return mapping
    
    def clean_and_normalize(self, df):
        """Clean and normalize PDF-extracted data using existing CSV parser logic."""
        # Use the existing CSV parser's cleaning methods
        csv_parser = BankStatementParser()
        
        # Handle separate Debit/Credit columns
        if 'Debit' in df.columns or 'Credit' in df.columns:
            df = self._combine_debit_credit_columns(df)
        
        # Apply existing cleaning logic
        cleaned_df = csv_parser.clean_and_normalize(df)
        
        self.cleaned_data = cleaned_df
        return cleaned_df
    
    def _combine_debit_credit_columns(self, df):
        """Combine separate Debit and Credit columns into a single Amount column."""
        df = df.copy()
        
        # Initialize Amount column
        df['Amount'] = 0.0
        
        # Process Debit column (negative amounts)
        if 'Debit' in df.columns:
            debit_values = pd.to_numeric(df['Debit'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce').fillna(0)
            df['Amount'] -= debit_values
        
        # Process Credit column (positive amounts)
        if 'Credit' in df.columns:
            credit_values = pd.to_numeric(df['Credit'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce').fillna(0)
            df['Amount'] += credit_values
        
        # Drop the original Debit/Credit columns
        df = df.drop(columns=[col for col in ['Debit', 'Credit'] if col in df.columns])
        
        return df
    
    def get_transaction_summary(self):
        """Get summary statistics of the parsed transactions."""
        if self.cleaned_data is None:
            return None
        
        df = self.cleaned_data
        
        # Safe date formatting
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'Unknown',
                'end': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'Unknown'
            },
            'total_debits': df[df['Amount'] < 0]['Amount'].sum(),
            'total_credits': df[df['Amount'] > 0]['Amount'].sum(),
            'net_change': df['Amount'].sum(),
            'largest_expense': df[df['Amount'] < 0]['Amount'].min() if len(df[df['Amount'] < 0]) > 0 else 0,
            'largest_income': df[df['Amount'] > 0]['Amount'].max() if len(df[df['Amount'] > 0]) > 0 else 0
        }
        
        return summary

def parse_bank_statement(file_path):
    """Convenience function to parse a bank statement CSV."""
    parser = BankStatementParser()
    raw_data = parser.load_csv(file_path)
    cleaned_data = parser.clean_and_normalize(raw_data)
    summary = parser.get_transaction_summary()
    
    return cleaned_data, summary

def pdf_to_csv_converter(file_path_or_buffer):
    """Convert PDF bank statement to CSV format, then parse with existing CSV parser."""
    import tempfile
    import os
    
    try:
        # Extract tabular data from PDF
        csv_data = extract_tables_from_pdf_to_csv(file_path_or_buffer)
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(csv_data)
            temp_csv_path = temp_file.name
        
        try:
            # Use our existing proven CSV parser
            cleaned_data, summary = parse_bank_statement(temp_csv_path)
            return cleaned_data, summary
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                
    except Exception as e:
        raise ValueError(f"Error converting PDF to CSV: {str(e)}")

def extract_tables_from_pdf_to_csv(file_path_or_buffer):
    """Extract transaction tables from PDF and convert to CSV format."""
    try:
        if isinstance(file_path_or_buffer, (str, bytes)):
            with pdfplumber.open(file_path_or_buffer) as pdf:
                return _extract_and_convert_to_csv(pdf)
        else:
            with pdfplumber.open(file_path_or_buffer) as pdf:
                return _extract_and_convert_to_csv(pdf)
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def _extract_and_convert_to_csv(pdf):
    """Extract tables from PDF and convert to CSV string."""
    import csv
    from io import StringIO
    
    # Collect all transaction tables across all pages
    all_transaction_tables = []
    
    for page_num, page in enumerate(pdf.pages):
        try:
            tables = page.extract_tables()
            if not tables:
                continue
                
            for table in tables:
                if not table or len(table) < 2:  # Need at least header + 1 data row
                    continue
                
                # Score this table based on how likely it is to be a transaction table
                score = _score_transaction_table(table)
                if score >= 3:  # Accept tables with decent scores
                    all_transaction_tables.append((table, score, page_num))
                    
        except Exception as e:
            print(f"Warning: Error processing page {page_num + 1}: {str(e)}")
            continue
    
    # Debug: Show all table scores
    print(f"DEBUG: Found {len([t for page in pdf.pages for t in (page.extract_tables() or []) if t])} tables total")
    print(f"DEBUG: Found {len(all_transaction_tables)} transaction tables")
    
    if not all_transaction_tables:
        raise ValueError(f"No suitable transaction tables found in PDF.")
    
    # Sort by score and take only the best table per page to avoid duplicates
    all_transaction_tables.sort(key=lambda x: x[1], reverse=True)
    
    # Take only the highest-scoring table per page
    best_tables_per_page = {}
    for table, score, page_num in all_transaction_tables:
        if page_num not in best_tables_per_page or score > best_tables_per_page[page_num][1]:
            best_tables_per_page[page_num] = (table, score, page_num)
    
    # Convert back to list
    selected_tables = list(best_tables_per_page.values())
    print(f"DEBUG: Selected {len(selected_tables)} best tables (one per page)")
    
    # Process the selected FNB-style tables and reconstruct proper transactions
    return _process_fnb_style_tables(selected_tables)

def _process_fnb_style_tables(transaction_tables):
    """Process FNB-style tables where data is spread across multiple rows."""
    import csv
    from io import StringIO
    
    all_transactions = []
    
    for table, score, page_num in transaction_tables:
        print(f"DEBUG: Processing table from page {page_num + 1} with score {score}")
        
        # Check if this is an FNB-style table (amounts in one row, dates in separate rows)
        if _is_fnb_style_table(table):
            transactions = _extract_fnb_transactions(table)
            all_transactions.extend(transactions)
            print(f"DEBUG: Extracted {len(transactions)} transactions from FNB-style table")
        else:
            # Process as regular table
            transactions = _extract_regular_transactions(table)
            all_transactions.extend(transactions)
            print(f"DEBUG: Extracted {len(transactions)} transactions from regular table")
    
    # Remove duplicates based on date, description, and amount
    unique_transactions = []
    seen = set()
    
    for transaction in all_transactions:
        # Create a unique key for each transaction
        key = (
            transaction.get('Date', '').strip(),
            transaction.get('Description', '').strip()[:50],  # First 50 chars to handle minor variations
            transaction.get('Amount', '').strip()
        )
        
        # Must have date and amount, and amount must not be zero
        if key not in seen and key[0] and key[2] and abs(float(key[2])) >= 0.01:
            seen.add(key)
            unique_transactions.append(transaction)
        else:
            if key in seen:
                print(f"DEBUG: Skipping duplicate transaction: {key[0]} - {key[1][:30]}... - {key[2]}")
            elif not key[2] or abs(float(key[2])) < 0.01:
                print(f"DEBUG: Skipping zero amount transaction: {key[0]} - {key[1][:30]}...")
    
    print(f"DEBUG: Removed {len(all_transactions) - len(unique_transactions)} duplicates")
    
    # Convert to CSV
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)
    
    # Write header
    csv_writer.writerow(['Date', 'Description', 'Amount', 'Balance'])
    
    # Write unique transactions
    for transaction in unique_transactions:
        csv_writer.writerow([
            transaction.get('Date', ''),
            transaction.get('Description', ''),
            transaction.get('Amount', ''),
            transaction.get('Balance', '')
        ])
    
    csv_data = csv_output.getvalue()
    print(f"DEBUG: Generated CSV with {len(unique_transactions)} unique transactions, {len(csv_data)} characters")
    return csv_data

def _is_fnb_style_table(table):
    """Check if this is an FNB-style table with amounts in one row."""
    if len(table) < 3:
        return False
    
    # Look for a row with many amounts separated by spaces/newlines
    for row in table[1:3]:  # Check first few data rows
        for cell in row:
            if cell and isinstance(cell, str):
                # Count number patterns (amounts) in the cell
                import re
                amount_patterns = re.findall(r'\d+\.?\d*', cell.replace(',', ''))
                if len(amount_patterns) > 5:  # If many amounts in one cell
                    return True
    return False

def _extract_fnb_transactions(table):
    """Extract transactions from FNB-style table."""
    transactions = []
    
    # Find the amounts row (usually row 1 or 2)
    amounts_row = None
    balances_row = None
    
    for i, row in enumerate(table[1:4]):  # Check first few data rows
        for j, cell in enumerate(row):
            if cell and isinstance(cell, str):
                # Look for amounts (many numbers)
                import re
                # More flexible pattern to catch amounts
                cell_clean = cell.replace(',', '').replace(' ', '\n').replace('\r', '\n')
                numbers = re.findall(r'\d+\.?\d*(?:C)?', cell_clean)
                if len(numbers) > 5:
                    if 'C' in cell:  # Credits/balances
                        balances_row = (i + 1, j, numbers)
                        print(f"DEBUG: Found balances row at {i+1},{j} with {len(numbers)} balances")
                    else:  # Amounts (debits)
                        amounts_row = (i + 1, j, numbers)
                        print(f"DEBUG: Found amounts row at {i+1},{j} with {len(numbers)} amounts")
                
                # Also check if this could be amounts without 'C' suffix
                if not amounts_row and len(numbers) > 5 and not any('C' in n for n in numbers):
                    # Extract just the numeric parts for amounts
                    clean_amounts = [re.sub(r'[^\d.]', '', n) for n in numbers if re.match(r'\d', n)]
                    if len(clean_amounts) > 5:
                        amounts_row = (i + 1, j, clean_amounts)
                        print(f"DEBUG: Found amounts row (no C) at {i+1},{j} with {len(clean_amounts)} amounts")
    
    # Find date rows (rows with dates)
    date_rows = []
    for i, row in enumerate(table[2:]):  # Skip header and amounts row
        if row and len(row) > 0 and row[0]:
            date_str = str(row[0]).strip()
            # Check if it looks like a date
            import re
            if re.match(r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', date_str):
                description = row[1] if len(row) > 1 and row[1] else ''
                date_rows.append((date_str, description))
    
    print(f"DEBUG: Found {len(date_rows)} date rows, amounts_row: {amounts_row is not None}, balances_row: {balances_row is not None}")
    
    # Match dates with amounts and balances
    if date_rows and (amounts_row or balances_row):
        amounts = amounts_row[2] if amounts_row else []
        balances = balances_row[2] if balances_row else []
        
        # If we have balances but no amounts, calculate amounts from balance differences
        if not amounts and balances:
            amounts = []
            for i in range(len(balances) - 1):
                try:
                    current_balance = float(re.sub(r'[^\d.]', '', balances[i]))
                    next_balance = float(re.sub(r'[^\d.]', '', balances[i + 1]))
                    amount_diff = current_balance - next_balance
                    
                    # Only add if there's a meaningful difference (avoid tiny floating point errors)
                    if abs(amount_diff) >= 0.01:
                        amounts.append(str(amount_diff))  # Keep sign (positive = credit, negative = debit)
                    else:
                        amounts.append('0')
                except:
                    amounts.append('0')
            print(f"DEBUG: Calculated {len(amounts)} amounts from balance differences")
        
        # Create transactions by matching dates with amounts
        for i, (date, description) in enumerate(date_rows):
            if i < len(amounts) or i < len(balances):
                amount = amounts[i] if i < len(amounts) else ''
                balance = balances[i] if i < len(balances) else ''
                
                # Clean up the values
                if amount and str(amount) != '0':
                    try:
                        amount_float = float(str(amount))
                        # Round to 2 decimal places to avoid floating point issues
                        amount_float = round(amount_float, 2)
                        
                        # Skip zero amounts
                        if abs(amount_float) < 0.01:
                            continue
                        
                        # For calculated amounts from balance differences, the sign is already correct
                        if not amounts_row:  # Calculated amounts
                            amount = str(-amount_float)  # Invert sign (balance decrease = expense)
                        else:  # Direct amounts from PDF
                            if str(amount).endswith('C'):
                                amount = str(amount_float)  # Credit (positive)
                            else:
                                amount = str(-amount_float)  # Debit (negative)
                    except:
                        continue  # Skip invalid amounts
                
                transactions.append({
                    'Date': date,
                    'Description': description,
                    'Amount': amount,
                    'Balance': balance.replace('C', '') if balance else ''
                })
    
    return transactions

def _extract_regular_transactions(table):
    """Extract transactions from regular table format."""
    transactions = []
    
    # Find header row
    header_row = None
    for i, row in enumerate(table):
        if row and any('date' in str(cell).lower() for cell in row if cell):
            header_row = i
            break
    
    if header_row is None:
        return transactions
    
    # Map columns
    headers = table[header_row]
    column_map = {}
    for i, header in enumerate(headers):
        if header:
            header_lower = str(header).lower()
            if 'date' in header_lower:
                column_map['Date'] = i
            elif 'description' in header_lower:
                column_map['Description'] = i
            elif 'amount' in header_lower:
                column_map['Amount'] = i
            elif 'balance' in header_lower:
                column_map['Balance'] = i
    
    # Extract transactions
    for row in table[header_row + 1:]:
        if row and any(cell for cell in row):
            transaction = {}
            for field, col_idx in column_map.items():
                if col_idx < len(row) and row[col_idx]:
                    transaction[field] = str(row[col_idx]).strip()
            
            if transaction.get('Date') and transaction.get('Amount'):
                transactions.append(transaction)
    
    return transactions

def _score_transaction_table(table):
    """Score a table based on how likely it is to be a transaction table."""
    if not table or len(table) < 2:
        return 0
    
    score = 0
    
    # Check for transaction-related headers in first few rows
    header_indicators = [
        'date', 'description', 'amount', 'balance', 'debit', 'credit',
        'transaction', 'detail', 'narrative', 'reference', 'value',
        'withdrawal', 'deposit', 'payment', 'transfer'
    ]
    
    # Look at first 3 rows for headers
    for row_idx in range(min(3, len(table))):
        row = table[row_idx]
        if not row:
            continue
            
        row_text = ' '.join([str(cell).lower().strip() if cell else '' for cell in row])
        
        # Count header matches
        matches = sum(1 for indicator in header_indicators if indicator in row_text)
        score += matches * 2  # Weight header matches heavily
        
        # Bonus for having the right number of columns (4-8 is typical)
        if 4 <= len([cell for cell in row if cell is not None]) <= 8:
            score += 1
    
    # Bonus for having enough data rows
    if len(table) > 10:  # Good number of transactions
        score += 2
    elif len(table) > 5:
        score += 1
    
    # Look for date patterns in the data
    for row_idx in range(1, min(6, len(table))):  # Check first 5 data rows
        row = table[row_idx]
        if not row:
            continue
            
        for cell in row:
            if cell and isinstance(cell, str):
                # Look for date patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)
                import re
                date_patterns = [
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
                    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
                    r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Mon
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, cell, re.IGNORECASE):
                        score += 1
                        break
    
    return score

def parse_pdf_bank_statement(file_path_or_buffer):
    """Convenience function to parse a bank statement PDF using PDF-to-CSV conversion."""
    try:
        # Use the new PDF-to-CSV approach
        return pdf_to_csv_converter(file_path_or_buffer)
    except Exception as e:
        # If PDF-to-CSV fails, provide helpful error message
        raise ValueError(f"Error converting PDF to CSV format: {str(e)}. Please ensure your PDF contains a clear transaction table with columns like Date, Description, Amount, and Balance.")

if __name__ == "__main__":
    # Test with sample data
    try:
        df, summary = parse_bank_statement("sample_data/sample_bank_statement.csv")
        print("Successfully parsed bank statement!")
        print(f"Summary: {summary}")
        print(f"First 5 transactions:")
        print(df.head())
    except Exception as e:
        print(f"Error parsing bank statement: {e}")