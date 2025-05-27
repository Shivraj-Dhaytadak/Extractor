"""
aws_service_ingest.py

A standalone script to ingest all CSV files in a directory into individual SQL tables.
Supports SQLite and PostgreSQL via SQLAlchemy.

Usage:
    python aws_service_ingest.py --csv-dir ./csv_files --db-url sqlite:///services.db

Requirements:
    pip install sqlalchemy psycopg2-binary  # for Postgres
"""
import os
import glob
import csv
import re
import argparse
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Numeric, DateTime


def infer_column(name, sample_value):
    """Infer SQLAlchemy Column type from a sample value."""
    if sample_value in (None, ''):
        return Column(name, String)
    try:
        int(sample_value)
        return Column(name, Integer)
    except (ValueError, TypeError):
        pass
    try:
        float(sample_value)
        return Column(name, Numeric)
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            datetime.strptime(sample_value, fmt)
            return Column(name, DateTime)
        except (ValueError, TypeError):
            continue
    return Column(name, String)


def sanitize_table_name(filename):
    """Sanitize filename to valid SQL table name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', base)
    if re.match(r'^\d', safe):
        safe = f"t_{safe}"
    return safe.lower()


def sanitize_column_name(name):
    """Sanitize column name to lowercase with underscores."""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    if re.match(r'^\d', safe):
        safe = f"col_{safe}"
    return safe.lower()


def parse_date(value):
    """Parse a date string into a datetime object."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


class AWSServiceDB:
    def __init__(self, db_url):
        """Initialize database connection and metadata."""
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self._date_cols = {}
        self._num_cols = {}
        self._int_cols = {}

    def create_table(self, table_name, raw_headers, sample_row):
        """Create a table dynamically based on headers and a sample row."""
        cols = [Column('id', Integer, primary_key=True, autoincrement=True)]
        headers = [sanitize_column_name(h) for h in raw_headers]
        for header, sample in zip(headers, sample_row):
            cols.append(infer_column(header, sample))
        table = Table(table_name, self.metadata, *cols)
        table.create(bind=self.engine, checkfirst=True)

        self._date_cols[table_name] = [c.name for c in table.columns if isinstance(c.type, DateTime)]
        self._num_cols[table_name] = [c.name for c in table.columns if isinstance(c.type, Numeric)]
        self._int_cols[table_name] = [c.name for c in table.columns if isinstance(c.type, Integer) and c.name != 'id']
        return table

    def ingest_directory(self, csv_dir):
        """Ingest all CSVs in the given directory into individual tables."""
        files = glob.glob(os.path.join(csv_dir, '*.csv'))
        if not files:
            print(f"No CSV files found in {csv_dir}")
            return
        for path in files:
            table_name = sanitize_table_name(path)
            print(f"Ingesting '{path}' -> table '{table_name}'")
            with open(path, newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                raw_headers = next(reader, None)
                if not raw_headers:
                    print(f"  Skipped empty file: {path}")
                    continue
                rows = list(reader)
                if not rows:
                    print(f"  Skipped no-data file: {path}")
                    continue

                headers = [sanitize_column_name(h) for h in raw_headers]
                sample_row = next((r for r in rows if any(r)), [''] * len(headers))
                table = self.create_table(table_name, raw_headers, sample_row)

                date_cols = self._date_cols.get(table_name, [])
                num_cols = self._num_cols.get(table_name, [])
                int_cols = self._int_cols.get(table_name, [])

                records = []
                for row in rows:
                    row = row[:len(headers)] + [''] * (len(headers) - len(row))  # Normalize row length
                    rec = dict(zip(headers, row))

                    for col in date_cols:
                        val = rec.get(col, '')
                        rec[col] = parse_date(val) if val else None

                    for col in num_cols:
                        val = rec.get(col, '')
                        if isinstance(val, str) and val.lower() == 'inf':
                            rec[col] = float('inf')
                        elif val == '':
                            rec[col] = None
                        else:
                            try:
                                rec[col] = float(val)
                            except (ValueError, TypeError):
                                rec[col] = None

                    for col in int_cols:
                        val = rec.get(col, '')
                        if isinstance(val, str) and val.lower() == 'inf':
                            rec[col] = float('inf')
                        elif val == '':
                            rec[col] = None
                        else:
                            try:
                                rec[col] = int(val)
                            except (ValueError, TypeError):
                                try:
                                    rec[col] = int(float(val))
                                except (ValueError, TypeError):
                                    rec[col] = None
                    records.append(rec)

                with self.engine.begin() as conn:
                    conn.execute(table.insert(), records)
                print(f"  Loaded {len(records)} rows into '{table_name}'")


def parse_args():
    parser = argparse.ArgumentParser(description='Ingest AWS service CSVs into SQL tables')
    parser.add_argument('--csv-dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--db-url', default='sqlite:///services.db',
                        help='Database URL (e.g., sqlite:///services.db or postgresql://user:pass@host/db)')
    return parser.parse_args()


def main():
    db = AWSServiceDB('sqlite:///services.db')
    db.ingest_directory(r'C:\VScodeMaster\Inferencing\Extractor\experiments\csv')


if __name__ == '__main__':
    main()
