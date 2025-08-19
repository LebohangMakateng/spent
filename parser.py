"""
CSV Parser for Bank Statements - Spent Money Detective
Handles loading, cleaning, and normalizing bank statement data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

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
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        return df
    
    def _clean_amounts(self, df):
        """Clean and standardize amount formats."""
        # Convert Amount to numeric, handling various formats
        df['Amount'] = df['Amount'].astype(str)
        df['Amount'] = df['Amount'].str.replace(',', '')  # Remove commas
        df['Amount'] = df['Amount'].str.replace(' ', '')   # Remove spaces
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Clean Balance column similarly
        if 'Balance' in df.columns:
            df['Balance'] = df['Balance'].astype(str)
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
        
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'total_debits': df[df['Amount'] < 0]['Amount'].sum(),
            'total_credits': df[df['Amount'] > 0]['Amount'].sum(),
            'net_change': df['Amount'].sum(),
            'largest_expense': df[df['Amount'] < 0]['Amount'].min(),
            'largest_income': df[df['Amount'] > 0]['Amount'].max()
        }
        
        return summary

def parse_bank_statement(file_path):
    """Convenience function to parse a bank statement CSV."""
    parser = BankStatementParser()
    raw_data = parser.load_csv(file_path)
    cleaned_data = parser.clean_and_normalize(raw_data)
    summary = parser.get_transaction_summary()
    
    return cleaned_data, summary

if __name__ == "__main__":
    # Test with sample data
    try:
        df, summary = parse_bank_statement("sample_data/real_bank_statement.csv")
        print("Successfully parsed bank statement!")
        print(f"Summary: {summary}")
        print(f"First 5 transactions:")
        print(df.head())
    except Exception as e:
        print(f"Error parsing bank statement: {e}")