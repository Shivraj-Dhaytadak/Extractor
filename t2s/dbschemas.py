#!/usr/bin/env python3
"""
Simple XLSX to SQL Converter
Converts 3 XLSX files directly to 3 SQL tables
"""

import pandas as pd
import sqlite3
import os

class SimpleXLSXConverter:
    def __init__(self, db_path='financial_data.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect_db(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
    
    def create_schema(self):
        """Create database tables"""
        schema_sql = """
        -- Table 1: Main Financial Data
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            month VARCHAR(7),
            version VARCHAR(20),
            scenario VARCHAR(50),
            currency VARCHAR(3),
            entity VARCHAR(255),
            gl_account VARCHAR(50),
            job_assignment VARCHAR(20),
            location VARCHAR(255),
            property VARCHAR(255),
            department VARCHAR(255),
            measure VARCHAR(20),
            value DECIMAL(18,2)
        );

        -- Table 2: Entity Business Units
        CREATE TABLE IF NOT EXISTS entity_business_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity VARCHAR(255),
            business_unit VARCHAR(255),
            additional_mapping VARCHAR(255)
        );

        -- Table 3: GL Account Details
        CREATE TABLE IF NOT EXISTS gl_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gl_account VARCHAR(50),
            gl_description TEXT,
            pl_main_category VARCHAR(255),
            pl_sub_category VARCHAR(255)
        );

        -- Basic indexes
        CREATE INDEX IF NOT EXISTS idx_financial_data_entity ON financial_data(entity);
        CREATE INDEX IF NOT EXISTS idx_financial_data_gl_account ON financial_data(gl_account);
        """
        
        self.conn.executescript(schema_sql)
        self.conn.commit()
        print("Database schema created")
    
    def load_financial_data(self, file_path):
        """Load first XLSX file to financial_data table"""
        df = pd.read_excel(file_path)
        
        # Map columns to database fields
        column_mapping = {
            'Year': 'year',
            'Month': 'month',
            'Version': 'version',
            'Scenario': 'scenario',
            'Currency': 'currency',
            'Entity': 'entity',
            'GL Account': 'gl_account',
            'Job Assignment': 'job_assignment',
            'Location': 'location',
            'Property': 'property',
            'Department': 'department',
            'Measure': 'measure',
            'Value': 'value'
        }
        
        # Rename columns to match database
        df.rename(columns=column_mapping, inplace=True)
        
        # Insert data
        df.to_sql('financial_data', self.conn, if_exists='append', index=False)
        print(f"Loaded {len(df)} rows into financial_data table")
    
    def load_entity_business_units(self, file_path):
        """Load second XLSX file to entity_business_units table"""
        df = pd.read_excel(file_path)
        
        # Map columns to database fields
        column_mapping = {
            'Entity': 'entity',
            'Business Unit': 'business_unit',
            'Additional Mapping': 'additional_mapping'
        }
        
        # Rename columns to match database
        df.rename(columns=column_mapping, inplace=True)
        
        # Insert data
        df.to_sql('entity_business_units', self.conn, if_exists='append', index=False)
        print(f"Loaded {len(df)} rows into entity_business_units table")
    
    def load_gl_accounts(self, file_path):
        """Load third XLSX file to gl_accounts table"""
        df = pd.read_excel(file_path)
        
        # Map columns to database fields
        column_mapping = {
            'GL Account': 'gl_account',
            'GL Description': 'gl_description',
            'P&L Main Category': 'pl_main_category',
            'P&L Sub Category': 'pl_sub_category'
        }
        
        # Rename columns to match database
        df.rename(columns=column_mapping, inplace=True)
        
        # Insert data
        df.to_sql('gl_accounts', self.conn, if_exists='append', index=False)
        print(f"Loaded {len(df)} rows into gl_accounts table")
    
    def export_to_sql_file(self, output_file='financial_data_export.sql'):
        """Export database to SQL file"""
        with open(output_file, 'w') as f:
            for line in self.conn.iterdump():
                f.write('%s\n' % line)
        print(f"Database exported to {output_file}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main conversion function"""
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    financial_file = "financial_data.xlsx"      # First XLSX file
    entity_business_file = "entity_business.xlsx"  # Second XLSX file
    gl_account_file = "gl_accounts.xlsx"           # Third XLSX file
    
    converter = SimpleXLSXConverter('financial_data.db')
    
    try:
        converter.connect_db()
        converter.create_schema()
        
        # Load each XLSX file
        if os.path.exists(financial_file):
            converter.load_financial_data(financial_file)
        
        if os.path.exists(entity_business_file):
            converter.load_entity_business_units(entity_business_file)
        
        if os.path.exists(gl_account_file):
            converter.load_gl_accounts(gl_account_file)
        
        # Export to SQL file
        converter.export_to_sql_file()
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        converter.close()

if __name__ == "__main__":
    main()