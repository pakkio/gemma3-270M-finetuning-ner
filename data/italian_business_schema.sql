-- Italian Business Database Schema for Text-to-SQL Fine-tuning
-- Comprehensive schema covering typical Italian business scenarios

-- Clienti (Customers)
CREATE TABLE clienti (
    id_cliente INT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    cognome VARCHAR(100) NOT NULL,
    email VARCHAR(150),
    telefono VARCHAR(20),
    data_nascita DATE,
    citta VARCHAR(100),
    regione VARCHAR(50),
    cap VARCHAR(10),
    indirizzo VARCHAR(200),
    tipo_cliente VARCHAR(50), -- 'privato', 'azienda', 'ente_pubblico'
    data_registrazione DATE,
    stato VARCHAR(20) DEFAULT 'attivo' -- 'attivo', 'inattivo', 'sospeso'
);

-- Prodotti (Products)
CREATE TABLE prodotti (
    id_prodotto INT PRIMARY KEY,
    nome VARCHAR(200) NOT NULL,
    descrizione TEXT,
    categoria VARCHAR(100), -- 'elettronica', 'abbigliamento', 'casa', 'sport', 'libri'
    sottocategoria VARCHAR(100),
    prezzo DECIMAL(10,2) NOT NULL,
    costo DECIMAL(10,2),
    fornitore VARCHAR(150),
    giacenza INT DEFAULT 0,
    data_inserimento DATE,
    stato VARCHAR(20) DEFAULT 'disponibile' -- 'disponibile', 'esaurito', 'discontinuo'
);

-- Ordini (Orders)
CREATE TABLE ordini (
    id_ordine INT PRIMARY KEY,
    id_cliente INT,
    data_ordine DATE NOT NULL,
    data_consegna DATE,
    totale DECIMAL(12,2) NOT NULL,
    stato VARCHAR(30), -- 'in_elaborazione', 'spedito', 'consegnato', 'annullato', 'rimborsato'
    metodo_pagamento VARCHAR(50), -- 'carta_credito', 'bonifico', 'paypal', 'contrassegno'
    spese_spedizione DECIMAL(8,2) DEFAULT 0,
    citta_consegna VARCHAR(100),
    regione_consegna VARCHAR(50),
    note TEXT,
    FOREIGN KEY (id_cliente) REFERENCES clienti(id_cliente)
);

-- Dettagli Ordine (Order Details)
CREATE TABLE dettagli_ordine (
    id_dettaglio INT PRIMARY KEY,
    id_ordine INT,
    id_prodotto INT,
    quantita INT NOT NULL,
    prezzo_unitario DECIMAL(10,2) NOT NULL,
    sconto DECIMAL(5,2) DEFAULT 0,
    subtotale DECIMAL(12,2) NOT NULL,
    FOREIGN KEY (id_ordine) REFERENCES ordini(id_ordine),
    FOREIGN KEY (id_prodotto) REFERENCES prodotti(id_prodotto)
);

-- Dipendenti (Employees)
CREATE TABLE dipendenti (
    id_dipendente INT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    cognome VARCHAR(100) NOT NULL,
    email VARCHAR(150),
    telefono VARCHAR(20),
    posizione VARCHAR(100), -- 'manager', 'venditore', 'magazziniere', 'amministrativo'
    dipartimento VARCHAR(100), -- 'vendite', 'marketing', 'logistica', 'amministrazione'
    stipendio DECIMAL(10,2),
    data_assunzione DATE,
    citta VARCHAR(100),
    manager_id INT,
    stato VARCHAR(20) DEFAULT 'attivo', -- 'attivo', 'inattivo', 'licenziato'
    FOREIGN KEY (manager_id) REFERENCES dipendenti(id_dipendente)
);

-- Fornitori (Suppliers)
CREATE TABLE fornitori (
    id_fornitore INT PRIMARY KEY,
    nome_azienda VARCHAR(200) NOT NULL,
    contatto VARCHAR(100),
    email VARCHAR(150),
    telefono VARCHAR(20),
    citta VARCHAR(100),
    regione VARCHAR(50),
    paese VARCHAR(50) DEFAULT 'Italia',
    settore VARCHAR(100), -- 'tecnologia', 'tessile', 'alimentare', 'automotive'
    valutazione DECIMAL(3,2), -- da 1.00 a 5.00
    data_partnership DATE
);

-- Fatture (Invoices)
CREATE TABLE fatture (
    id_fattura INT PRIMARY KEY,
    numero_fattura VARCHAR(50) UNIQUE NOT NULL,
    id_cliente INT,
    id_ordine INT,
    data_emissione DATE NOT NULL,
    data_scadenza DATE,
    data_pagamento DATE,
    importo_totale DECIMAL(12,2) NOT NULL,
    iva DECIMAL(8,2),
    stato VARCHAR(30), -- 'emessa', 'pagata', 'scaduta', 'stornata'
    note TEXT,
    FOREIGN KEY (id_cliente) REFERENCES clienti(id_cliente),
    FOREIGN KEY (id_ordine) REFERENCES ordini(id_ordine)
);

-- Magazzino (Warehouse)
CREATE TABLE magazzino (
    id_movimento INT PRIMARY KEY,
    id_prodotto INT,
    tipo_movimento VARCHAR(20), -- 'entrata', 'uscita', 'inventario'
    quantita INT NOT NULL,
    data_movimento DATE NOT NULL,
    motivo VARCHAR(100), -- 'vendita', 'acquisto', 'reso', 'danneggiamento'
    note TEXT,
    id_dipendente INT,
    FOREIGN KEY (id_prodotto) REFERENCES prodotti(id_prodotto),
    FOREIGN KEY (id_dipendente) REFERENCES dipendenti(id_dipendente)
);

-- Promozioni (Promotions)
CREATE TABLE promozioni (
    id_promozione INT PRIMARY KEY,
    nome VARCHAR(200) NOT NULL,
    descrizione TEXT,
    sconto_percentuale DECIMAL(5,2),
    sconto_fisso DECIMAL(10,2),
    data_inizio DATE NOT NULL,
    data_fine DATE NOT NULL,
    categoria_prodotti VARCHAR(100),
    importo_minimo DECIMAL(10,2),
    stato VARCHAR(20) DEFAULT 'attiva' -- 'attiva', 'scaduta', 'sospesa'
);

-- Recensioni (Reviews)
CREATE TABLE recensioni (
    id_recensione INT PRIMARY KEY,
    id_cliente INT,
    id_prodotto INT,
    valutazione INT CHECK (valutazione >= 1 AND valutazione <= 5),
    titolo VARCHAR(200),
    commento TEXT,
    data_recensione DATE NOT NULL,
    verificata BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (id_cliente) REFERENCES clienti(id_cliente),
    FOREIGN KEY (id_prodotto) REFERENCES prodotti(id_prodotto)
);

-- Sample Italian cities and regions for realistic data
-- Common Italian cities: Milano, Roma, Napoli, Torino, Palermo, Genova, Bologna, Firenze, Bari, Catania
-- Italian regions: Lombardia, Lazio, Campania, Piemonte, Sicilia, Liguria, Emilia-Romagna, Toscana, Puglia