BEGIN TRANSACTION;
CREATE TABLE entity_business_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity VARCHAR(255),
            business_unit VARCHAR(255),
            additional_mapping VARCHAR(255)
        );
CREATE TABLE financial_data (
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
CREATE TABLE gl_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gl_account VARCHAR(50),
            gl_description TEXT,
            pl_main_category VARCHAR(255),
            pl_sub_category VARCHAR(255)
        );
CREATE INDEX idx_financial_data_entity ON financial_data(entity);
CREATE INDEX idx_financial_data_gl_account ON financial_data(gl_account);
DELETE FROM "sqlite_sequence";
COMMIT;
