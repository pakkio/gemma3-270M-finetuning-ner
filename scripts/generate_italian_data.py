#!/usr/bin/env python3
"""
Script to generate additional Italian training examples for entity extraction.
Generates examples with people, dates, and places in Italian contexts.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Italian names and surnames
ITALIAN_NAMES = [
    "Marco", "Giulia", "Francesco", "Chiara", "Alessandro", "Federica", "Andrea", "Valentina",
    "Matteo", "Francesca", "Lorenzo", "Sara", "Davide", "Martina", "Simone", "Elena",
    "Luca", "Giorgia", "Gabriele", "Alessia", "Riccardo", "Sofia", "Tommaso", "Beatrice",
    "Nicola", "Camilla", "Federico", "Arianna", "Giovanni", "Ilaria", "Antonio", "Caterina",
    "Michele", "Veronica", "Stefano", "Silvia", "Paolo", "Anna", "Roberto", "Maria",
    "Fabio", "Laura", "Emanuele", "Roberta", "Daniele", "Paola", "Claudio", "Serena"
]

ITALIAN_SURNAMES = [
    "Rossi", "Ferrari", "Russo", "Bianchi", "Romano", "Gallo", "Conti", "De Luca",
    "Mancini", "Costa", "Giordano", "Ricci", "Lombardi", "Moretti", "Barbieri", "Fontana",
    "Santoro", "Mariani", "Rinaldi", "Caruso", "Ferrara", "Galli", "Martini", "Leone",
    "Longo", "Gentile", "Martinelli", "Vitale", "Lombardo", "Serra", "Coppola", "De Santis",
    "D'Angelo", "Marchetti", "Parisi", "Villa", "Conte", "Ferretti", "Fabbri", "Marini"
]

# Italian cities and famous places
ITALIAN_PLACES = [
    "Roma", "Milano", "Napoli", "Torino", "Palermo", "Genova", "Bologna", "Firenze",
    "Bari", "Catania", "Venezia", "Verona", "Messina", "Padova", "Trieste", "Brescia",
    "Taranto", "Prato", "Parma", "Modena", "Reggio Calabria", "Reggio Emilia", "Perugia",
    "Livorno", "Ravenna", "Cagliari", "Foggia", "Rimini", "Salerno", "Ferrara", "Sassari",
    "Latina", "Giugliano in Campania", "Monza", "Siracusa", "Pescara", "Bergamo", "Forlì",
    "Trento", "Vicenza", "Terni", "Bolzano", "Novara", "Piacenza", "Ancona", "Andria",
    "Arezzo", "Udine", "Cesena", "Lecce"
]

ITALIAN_LANDMARKS = [
    "Colosseo", "Torre di Pisa", "Duomo di Milano", "Ponte di Rialto", "Fontana di Trevi",
    "Palazzo Ducale", "Teatro La Scala", "Galleria degli Uffizi", "Palazzo Pitti",
    "Castel Sant'Angelo", "Pantheon", "Piazza San Marco", "Castello Sforzesco",
    "Palazzo Reale", "Teatro San Carlo", "Museo di Capodimonte", "Villa Borghese",
    "Terme di Caracalla", "Palazzo Altemps", "Galleria Borghese", "Musei Vaticani",
    "Cappella Sistina", "Basilica di San Pietro", "Arena di Verona", "Palazzo Te",
    "Reggia di Caserta", "Palazzo della Pilotta", "Torre del Mangia", "Palazzo Pubblico",
    "Loggia dei Lanzi", "Ponte Vecchio", "Palazzo Vecchio", "Basilica di Santa Croce",
    "Biblioteca Nazionale", "Centro Storico", "Pinacoteca di Brera", "Ca' Rezzonico",
    "Palazzo Grassi", "Peggy Guggenheim Collection", "Doge's Palace", "St. Mark's Basilica"
]

# Professional titles and roles
TITLES = ["Prof.", "Dott.", "Ing.", "Avv.", "Arch.", "Dr."]

# Event types and contexts
EVENT_TYPES = [
    "conferenza", "convegno", "simposio", "festival", "mostra", "esposizione",
    "cerimonia", "premiazione", "presentazione", "seminario", "workshop", "corso",
    "lezione", "dibattito", "tavola rotonda", "concerto", "spettacolo", "evento",
    "manifestazione", "rassegna", "fiera", "congresso", "assemblea", "riunione"
]

VENUES = [
    "presso", "al", "alla", "nel", "nella", "dello", "della", "degli", "delle"
]

def generate_random_date() -> str:
    """Generate a random Italian date string."""
    months = [
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
    ]
    
    # Generate date between 2023 and 2026
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    random_date = start_date + timedelta(days=random_days)
    
    return f"{random_date.day} {months[random_date.month - 1]} {random_date.year}"

def generate_person_name() -> str:
    """Generate a random Italian person name."""
    name = random.choice(ITALIAN_NAMES)
    surname = random.choice(ITALIAN_SURNAMES)
    
    # Sometimes add a title
    if random.random() < 0.3:
        title = random.choice(TITLES)
        return f"{title} {name} {surname}"
    return f"{name} {surname}"

def generate_place() -> str:
    """Generate a random Italian place."""
    if random.random() < 0.6:
        return random.choice(ITALIAN_LANDMARKS)
    else:
        return random.choice(ITALIAN_PLACES)

def create_training_example() -> Dict[str, Any]:
    """Create a single training example with Italian text and entities."""
    
    # Decide what entities to include
    include_people = random.random() < 0.4
    include_dates = random.random() < 0.7
    include_places = random.random() < 0.9
    
    people = []
    dates = []
    places = []
    
    # Generate entities
    if include_people:
        num_people = random.randint(1, 2)
        people = [generate_person_name() for _ in range(num_people)]
    
    if include_dates:
        num_dates = random.randint(1, 2)
        dates = [generate_random_date() for _ in range(num_dates)]
    
    if include_places:
        num_places = random.randint(1, 3)
        places = [generate_place() for _ in range(num_places)]
    
    # Generate document text
    event_type = random.choice(EVENT_TYPES)
    venue_prep = random.choice(VENUES)
    
    # Create different sentence patterns based on available entities
    document = ""
    
    if people and dates and places:
        patterns = [
            f"{people[0]} presenterà il suo lavoro presso {places[0]} il {dates[0]}.",
            f"La conferenza di {people[0]} si terrà {venue_prep} {places[0]} il {dates[0]}.",
            f"{people[0]} terrà una lezione {venue_prep} {places[0]} il {dates[0]}.",
            f"Il {event_type} con {people[0]} avrà luogo presso {places[0]} il {dates[0]}."
        ]
        document = random.choice(patterns)
    elif dates and places:
        patterns = [
            f"La {event_type} si terrà {venue_prep} {places[0]} il {dates[0]}.",
            f"Il {event_type} avrà luogo presso {', '.join(places[:2])} il {dates[0]}.",
            f"La cerimonia si svolgerà tra {', '.join(places)} il {dates[0]}.",
            f"Il festival internazionale si terrà {venue_prep} {places[0]} il {dates[0]}."
        ]
        document = random.choice(patterns)
    elif people and places:
        patterns = [
            f"{people[0]} presenterà il suo libro presso {places[0]}.",
            f"La conferenza di {people[0]} si terrà {venue_prep} {places[0]}.",
            f"{people[0]} terrà una lezione {venue_prep} {places[0]}.",
            f"Il seminario con {people[0]} avrà luogo presso {places[0]}."
        ]
        document = random.choice(patterns)
    elif people and dates:
        patterns = [
            f"{people[0]} presenterà il {dates[0]}.",
            f"La conferenza di {people[0]} è prevista per il {dates[0]}.",
            f"{people[0]} terrà una lezione il {dates[0]}.",
            f"L'incontro con {people[0]} si svolgerà il {dates[0]}."
        ]
        document = random.choice(patterns)
    elif places:
        patterns = [
            f"La {event_type} si terrà {venue_prep} {places[0]}.",
            f"Il {event_type} avrà luogo presso {', '.join(places[:2])}.",
            f"L'evento si svolgerà tra {', '.join(places)}.",
            f"La manifestazione si terrà {venue_prep} {places[0]}."
        ]
        document = random.choice(patterns)
    elif dates:
        patterns = [
            f"La {event_type} è prevista per il {dates[0]}.",
            f"L'evento si svolgerà il {dates[0]}.",
            f"La manifestazione avrà luogo il {dates[0]}.",
            f"Il {event_type} è fissato per il {dates[0]}."
        ]
        document = random.choice(patterns)
    else:
        document = f"La {event_type} si terrà presso la sede principale."
    
    # Create output JSON
    output = {
        "people": people,
        "dates": dates,
        "places": places
    }
    
    return {
        "document": document,
        "output": json.dumps(output, ensure_ascii=False)
    }

def generate_dataset(num_examples: int) -> List[Dict[str, Any]]:
    """Generate a dataset of Italian examples."""
    examples = []
    for _ in range(num_examples):
        example = create_training_example()
        examples.append(example)
    return examples

def main():
    """Generate additional Italian training data."""
    # We need to go from 35 to 120+, so generate about 90 new examples
    new_examples = generate_dataset(90)
    
    # Save new training examples
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_new.jsonl', 'w', encoding='utf-8') as f:
        for example in new_examples[:70]:  # 70 for training
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save new validation examples  
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_new.jsonl', 'w', encoding='utf-8') as f:
        for example in new_examples[70:]:  # 20 for validation
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(new_examples)} new Italian examples")
    print(f"Training examples: 70")
    print(f"Validation examples: 20")

if __name__ == "__main__":
    main()