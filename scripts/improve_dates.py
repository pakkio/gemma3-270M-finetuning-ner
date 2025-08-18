#!/usr/bin/env python3
"""
Script to improve date extraction by adding more diverse and challenging date examples.
"""

import json
import random
from datetime import datetime, timedelta

# More comprehensive date patterns for Italian
DATE_PATTERNS = [
    # Standard dates
    "{day} {month} {year}",
    "il {day} {month} {year}",
    
    # Date ranges
    "dal {day1} al {day2} {month} {year}",
    "dal {day1} {month1} al {day2} {month2} {year}",
    "dal {day1} al {day2} {month} {year}",
    
    # Partial dates
    "{day} {month}",
    "{month} {year}",
    "il {day} {month}",
    
    # Relative dates
    "lunedì {day} {month}",
    "martedì {day} {month}",
    "mercoledì {day} {month}",
    "giovedì {day} {month}",
    "venerdì {day} {month}",
    "sabato {day} {month}",
    "domenica {day} {month}",
    
    # Alternative formats
    "{day}/{month}/{year}",
    "{day}-{month}-{year}",
    "{day}.{month}.{year}",
    
    # Ordinal dates
    "il {day}° {month} {year}",
    "{day}° {month}",
]

ITALIAN_MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
]

ITALIAN_MONTHS_SHORT = [
    "gen", "feb", "mar", "apr", "mag", "giu",
    "lug", "ago", "set", "ott", "nov", "dic"
]

# Time expressions
TIME_EXPRESSIONS = [
    "alle ore {hour}:00",
    "alle {hour}:30",
    "alle ore {hour}",
    "dalle {hour1} alle {hour2}",
    "nella mattinata",
    "nel pomeriggio",
    "in serata",
    "di mattina",
    "di sera",
]

def generate_date_examples():
    """Generate additional examples focusing on date extraction."""
    examples = []
    
    # Example 1: Single clear date
    examples.extend([
        {
            "document": "La conferenza di informatica si terrà giovedì 14 marzo 2024 presso l'Università di Pisa.",
            "output": '{"people":[],"dates":["giovedì 14 marzo 2024"],"places":["Università di Pisa"]}'
        },
        {
            "document": "Il Dr. Matteo Rossi presenterà i risultati della ricerca il 22/05/2024.",
            "output": '{"people":["Matteo Rossi"],"dates":["22/05/2024"],"places":[]}'
        },
        {
            "document": "La mostra di arte contemporanea sarà aperta dal 5 al 20 novembre 2024.",
            "output": '{"people":[],"dates":["dal 5 al 20 novembre 2024"],"places":[]}'
        },
        {
            "document": "Roberto Baggio terrà una conferenza stampa venerdì 8 giugno alle ore 15:00.",
            "output": '{"people":["Roberto Baggio"],"dates":["venerdì 8 giugno alle ore 15:00"],"places":[]}'
        },
        {
            "document": "L'assemblea generale si svolgerà il 1° dicembre 2024 presso la sede centrale.",
            "output": '{"people":[],"dates":["1° dicembre 2024"],"places":["sede centrale"]}'
        },
        {
            "document": "Il corso di formazione inizierà lunedì 15 gennaio e terminerà venerdì 30 gennaio 2025.",
            "output": '{"people":[],"dates":["lunedì 15 gennaio","venerdì 30 gennaio 2025"],"places":[]}'
        },
        {
            "document": "La prof.ssa Maria Bianchi terrà il seminario il 10.04.2024 nell'Aula Magna.",
            "output": '{"people":["Maria Bianchi"],"dates":["10.04.2024"],"places":["Aula Magna"]}'
        },
        {
            "document": "Il festival del cinema si svolgerà dal 18 settembre al 2 ottobre 2024 a Venezia.",
            "output": '{"people":[],"dates":["dal 18 settembre al 2 ottobre 2024"],"places":["Venezia"]}'
        },
        {
            "document": "L'incontro è programmato per martedì 12 marzo alle 10:30.",
            "output": '{"people":[],"dates":["martedì 12 marzo alle 10:30"],"places":[]}'
        },
        {
            "document": "Giorgio Armani presenterà la nuova collezione il 20-02-2025 a Milano.",
            "output": '{"people":["Giorgio Armani"],"dates":["20-02-2025"],"places":["Milano"]}'
        },
        {
            "document": "La riunione del consiglio si terrà ogni primo lunedì del mese.",
            "output": '{"people":[],"dates":["ogni primo lunedì del mese"],"places":[]}'
        },
        {
            "document": "Il concerto di Laura Pausini è previsto per sabato 16 nov 2024.",
            "output": '{"people":["Laura Pausini"],"dates":["sabato 16 nov 2024"],"places":[]}'
        },
        {
            "document": "La celebrazione si svolgerà domenica 25 dicembre presso la Basilica di San Pietro.",
            "output": '{"people":[],"dates":["domenica 25 dicembre"],"places":["Basilica di San Pietro"]}'
        },
        {
            "document": "Il dott. Luca Monti farà la presentazione mercoledì 3/7/2024 alle ore 14:00.",
            "output": '{"people":["Luca Monti"],"dates":["mercoledì 3/7/2024 alle ore 14:00"],"places":[]}'
        },
        {
            "document": "L'evento culturale avrà luogo dal 1° al 15° giorno di maggio 2024.",
            "output": '{"people":[],"dates":["dal 1° al 15° giorno di maggio 2024"],"places":[]}'
        },
        {
            "document": "Claudio Baglioni si esibirà giovedì 11 aprile 2025 al Teatro dell'Opera.",
            "output": '{"people":["Claudio Baglioni"],"dates":["giovedì 11 aprile 2025"],"places":["Teatro dell\'Opera"]}'
        },
        {
            "document": "La sessione di esami inizierà il 15/06/2024 e terminerà il 30/06/2024.",
            "output": '{"people":[],"dates":["15/06/2024","30/06/2024"],"places":[]}'
        },
        {
            "document": "Il Prof. Andrea Camilleri terrà la lezione magistrale martedì 9 settembre.",
            "output": '{"people":["Andrea Camilleri"],"dates":["martedì 9 settembre"],"places":[]}'
        },
        {
            "document": "La fiera dell'artigianato si svolgerà nel weekend del 12-13 ottobre 2024.",
            "output": '{"people":[],"dates":["weekend del 12-13 ottobre 2024"],"places":[]}'
        },
        {
            "document": "Michele Placido dirigerà lo spettacolo che debutta venerdì 5.12.2024.",
            "output": '{"people":["Michele Placido"],"dates":["venerdì 5.12.2024"],"places":[]}'
        },
        # Multiple dates in one text
        {
            "document": "Il festival si terrà il 10 maggio, 15 maggio e 20 maggio 2024 in diverse location.",
            "output": '{"people":[],"dates":["10 maggio","15 maggio","20 maggio 2024"],"places":[]}'
        },
        {
            "document": "Elena Ferrante presenterà il libro il 3 marzo a Roma e il 5 marzo a Milano.",
            "output": '{"people":["Elena Ferrante"],"dates":["3 marzo","5 marzo"],"places":["Roma","Milano"]}'
        },
        # Complex temporal expressions
        {
            "document": "L'incontro è programmato per la prima settimana di aprile 2024.",
            "output": '{"people":[],"dates":["prima settimana di aprile 2024"],"places":[]}'
        },
        {
            "document": "La conferenza si svolgerà nella seconda metà di giugno.",
            "output": '{"people":[],"dates":["seconda metà di giugno"],"places":[]}'
        },
        {
            "document": "Il corso inizia l'ultima settimana di agosto 2024.",
            "output": '{"people":[],"dates":["ultima settimana di agosto 2024"],"places":[]}'
        },
        # Dates with no people or places
        {
            "document": "L'evento è rimandato al 15 ottobre 2024.",
            "output": '{"people":[],"dates":["15 ottobre 2024"],"places":[]}'
        },
        {
            "document": "La scadenza è fissata per il 31/12/2024.",
            "output": '{"people":[],"dates":["31/12/2024"],"places":[]}'
        },
        {
            "document": "Il termine ultimo è giovedì 28 novembre.",
            "output": '{"people":[],"dates":["giovedì 28 novembre"],"places":[]}'
        },
        # Time-only expressions
        {
            "document": "La riunione inizierà alle 9:30 del mattino.",
            "output": '{"people":[],"dates":["alle 9:30 del mattino"],"places":[]}'
        },
        {
            "document": "L'appuntamento è fissato per le ore 16:00.",
            "output": '{"people":[],"dates":["ore 16:00"],"places":[]}'
        },
        {
            "document": "Il pranzo si svolgerà dalle 12:30 alle 14:00.",
            "output": '{"people":[],"dates":["dalle 12:30 alle 14:00"],"places":[]}'
        }
    ])
    
    return examples

def main():
    """Add improved date examples to the dataset."""
    # Load existing data
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train.jsonl', 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val.jsonl', 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Generate new date-focused examples
    date_examples = generate_date_examples()
    
    # Add to training data (75% of new examples)
    train_addition = date_examples[:23]  # 23 examples for training
    val_addition = date_examples[23:]     # 7 examples for validation
    
    # Combine with existing data
    new_train_data = train_data + train_addition
    new_val_data = val_data + val_addition
    
    # Write updated datasets
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_date_improved.jsonl', 'w', encoding='utf-8') as f:
        for example in new_train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_date_improved.jsonl', 'w', encoding='utf-8') as f:
        for example in new_val_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Added {len(date_examples)} date-focused examples")
    print(f"New training set: {len(new_train_data)} examples")
    print(f"New validation set: {len(new_val_data)} examples")
    print(f"Total examples: {len(new_train_data) + len(new_val_data)}")

if __name__ == "__main__":
    main()