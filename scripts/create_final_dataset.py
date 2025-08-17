#!/usr/bin/env python3
"""
Create final expanded Italian dataset with 120+ examples for entity extraction.
"""

import json
import random
from datetime import datetime, timedelta

# High-quality Italian examples to add to the dataset
ADDITIONAL_EXAMPLES = [
    {
        "document": "Il Prof. Mario Draghi terrà una conferenza presso l'Università Bocconi di Milano il 15 marzo 2025.",
        "output": '{"people":["Mario Draghi"],"dates":["15 marzo 2025"],"places":["Università Bocconi di Milano"]}'
    },
    {
        "document": "La mostra di arte contemporanea si terrà al Palazzo Grassi di Venezia dal 10 aprile al 20 maggio 2024.",
        "output": '{"people":[],"dates":["10 aprile","20 maggio 2024"],"places":["Palazzo Grassi di Venezia"]}'
    },
    {
        "document": "Il festival del cinema si svolgerà presso il Teatro Romano di Verona il 22 settembre 2024.",
        "output": '{"people":[],"dates":["22 settembre 2024"],"places":["Teatro Romano di Verona"]}'
    },
    {
        "document": "La Dr.ssa Elena Bonetti presenterà il suo ultimo libro presso la Biblioteca Ambrosiana di Milano.",
        "output": '{"people":["Elena Bonetti"],"dates":[],"places":["Biblioteca Ambrosiana di Milano"]}'
    },
    {
        "document": "Il convegno internazionale di medicina si terrà presso l'Ospedale San Raffaele di Milano il 5 giugno 2024.",
        "output": '{"people":[],"dates":["5 giugno 2024"],"places":["Ospedale San Raffaele di Milano"]}'
    },
    {
        "document": "Roberto Benigni reciterà al Teatro dell'Opera di Roma il 18 dicembre 2024.",
        "output": '{"people":["Roberto Benigni"],"dates":["18 dicembre 2024"],"places":["Teatro dell\'Opera di Roma"]}'
    },
    {
        "document": "La cerimonia di laurea si svolgerà presso l'Aula Magna dell'Università La Sapienza di Roma.",
        "output": '{"people":[],"dates":[],"places":["Aula Magna dell\'Università La Sapienza di Roma"]}'
    },
    {
        "document": "Il Prof. Giuseppe Conte terrà una lezione magistrale presso l'Università di Firenze il 7 ottobre 2024.",
        "output": '{"people":["Giuseppe Conte"],"dates":["7 ottobre 2024"],"places":["Università di Firenze"]}'
    },
    {
        "document": "La Fiera dell'Antiquariato si terrà in Piazza Santo Spirito a Firenze dal 15 al 25 ottobre 2024.",
        "output": '{"people":[],"dates":["15 al 25 ottobre 2024"],"places":["Piazza Santo Spirito a Firenze"]}'
    },
    {
        "document": "Matteo Salvini parteciperà al dibattito politico presso la Sala dei Cinquecento di Palazzo Vecchio.",
        "output": '{"people":["Matteo Salvini"],"dates":[],"places":["Sala dei Cinquecento di Palazzo Vecchio"]}'
    },
    {
        "document": "Il concerto di Ludovico Einaudi avrà luogo presso l'Auditorium Parco della Musica di Roma il 30 novembre 2024.",
        "output": '{"people":["Ludovico Einaudi"],"dates":["30 novembre 2024"],"places":["Auditorium Parco della Musica di Roma"]}'
    },
    {
        "document": "La conferenza stampa si terrà presso la Sala Stampa del Quirinale a Roma alle ore 15:00.",
        "output": '{"people":[],"dates":[],"places":["Sala Stampa del Quirinale a Roma"]}'
    },
    {
        "document": "Il festival della letteratura vedrà la partecipazione di Alessandro Baricco presso il Salone del Libro di Torino.",
        "output": '{"people":["Alessandro Baricco"],"dates":[],"places":["Salone del Libro di Torino"]}'
    },
    {
        "document": "La mostra fotografica di Annie Leibovitz sarà esposta presso il MAXXI di Roma dal 12 gennaio al 15 marzo 2025.",
        "output": '{"people":["Annie Leibovitz"],"dates":["12 gennaio al 15 marzo 2025"],"places":["MAXXI di Roma"]}'
    },
    {
        "document": "Il seminario di economia si svolgerà presso la Camera di Commercio di Milano il 9 febbraio 2025.",
        "output": '{"people":[],"dates":["9 febbraio 2025"],"places":["Camera di Commercio di Milano"]}'
    },
    {
        "document": "Pier Luigi Bersani interverrà al convegno presso l'Università Cattolica del Sacro Cuore di Milano.",
        "output": '{"people":["Pier Luigi Bersani"],"dates":[],"places":["Università Cattolica del Sacro Cuore di Milano"]}'
    },
    {
        "document": "La premiazione del Premio Strega si terrà presso Villa Giulia a Roma il 6 luglio 2024.",
        "output": '{"people":[],"dates":["6 luglio 2024"],"places":["Villa Giulia a Roma"]}'
    },
    {
        "document": "Il Dr. Fabrizio Pregliasco terrà una conferenza sulla prevenzione sanitaria presso l'Istituto Superiore di Sanità.",
        "output": '{"people":["Fabrizio Pregliasco"],"dates":[],"places":["Istituto Superiore di Sanità"]}'
    },
    {
        "document": "La sfilata di moda si svolgerà presso Palazzo Pitti a Firenze durante la Fashion Week del gennaio 2025.",
        "output": '{"people":[],"dates":["gennaio 2025"],"places":["Palazzo Pitti a Firenze"]}'
    },
    {
        "document": "Il summit internazionale G20 avrà luogo presso il Centro Congressi EUR di Roma dal 14 al 16 ottobre 2024.",
        "output": '{"people":[],"dates":["14 al 16 ottobre 2024"],"places":["Centro Congressi EUR di Roma"]}'
    },
    {
        "document": "La chef Antonino Cannavacciuolo aprirà il nuovo ristorante presso il Castello di Rivoli il 20 marzo 2025.",
        "output": '{"people":["Antonino Cannavacciuolo"],"dates":["20 marzo 2025"],"places":["Castello di Rivoli"]}'
    },
    {
        "document": "Il corso di formazione per insegnanti si terrà presso l'Università di Bologna dal 5 al 10 settembre 2024.",
        "output": '{"people":[],"dates":["5 al 10 settembre 2024"],"places":["Università di Bologna"]}'
    },
    {
        "document": "Beppe Grillo terrà uno spettacolo comico presso il Teatro Ariston di Sanremo il 14 febbraio 2025.",
        "output": '{"people":["Beppe Grillo"],"dates":["14 febbraio 2025"],"places":["Teatro Ariston di Sanremo"]}'
    },
    {
        "document": "La conferenza internazionale di tecnologia si svolgerà presso il Politecnico di Torino.",
        "output": '{"people":[],"dates":[],"places":["Politecnico di Torino"]}'
    },
    {
        "document": "Il Prof. Umberto Eco terrà l'ultima lezione presso l'Università di Bologna il 19 febbraio 2016.",
        "output": '{"people":["Umberto Eco"],"dates":["19 febbraio 2016"],"places":["Università di Bologna"]}'
    },
    {
        "document": "La mostra di Leonardo da Vinci sarà ospitata presso gli Uffizi di Firenze fino al 31 dicembre 2024.",
        "output": '{"people":["Leonardo da Vinci"],"dates":["31 dicembre 2024"],"places":["Uffizi di Firenze"]}'
    },
    {
        "document": "Il festival jazz di Umbria si terrà presso l'Arena Santa Giuliana di Perugia dal 10 al 20 luglio 2024.",
        "output": '{"people":[],"dates":["10 al 20 luglio 2024"],"places":["Arena Santa Giuliana di Perugia"]}'
    },
    {
        "document": "Giorgio Napolitano inaugurerà la mostra presso il Palazzo del Quirinale a Roma.",
        "output": '{"people":["Giorgio Napolitano"],"dates":[],"places":["Palazzo del Quirinale a Roma"]}'
    },
    {
        "document": "La Biennale di Venezia 2024 si svolgerà presso i Giardini della Biennale dal 20 aprile al 24 novembre.",
        "output": '{"people":[],"dates":["20 aprile al 24 novembre"],"places":["Giardini della Biennale"]}'
    },
    {
        "document": "Il convegno medico vedrà la partecipazione del Prof. Paolo Veronesi presso l'Istituto Europeo di Oncologia di Milano.",
        "output": '{"people":["Paolo Veronesi"],"dates":[],"places":["Istituto Europeo di Oncologia di Milano"]}'
    },
    {
        "document": "La conferenza sul cambiamento climatico si terrà presso l'Università Ca' Foscari di Venezia il 22 aprile 2024.",
        "output": '{"people":[],"dates":["22 aprile 2024"],"places":["Università Ca\' Foscari di Venezia"]}'
    },
    {
        "document": "Andrea Bocelli si esibirà in concerto presso l'Arena di Verona il 15 agosto 2024.",
        "output": '{"people":["Andrea Bocelli"],"dates":["15 agosto 2024"],"places":["Arena di Verona"]}'
    },
    {
        "document": "Il workshop di fotografia si svolgerà presso la Fondazione Fotografia Modena dal 3 al 7 giugno 2024.",
        "output": '{"people":[],"dates":["3 al 7 giugno 2024"],"places":["Fondazione Fotografia Modena"]}'
    },
    {
        "document": "La conferenza di Renzo Piano sull'architettura sostenibile avrà luogo presso il Palazzo delle Esposizioni di Roma.",
        "output": '{"people":["Renzo Piano"],"dates":[],"places":["Palazzo delle Esposizioni di Roma"]}'
    },
    {
        "document": "Il festival del cinema di Roma si terrà presso l'Auditorium Parco della Musica dal 16 al 26 ottobre 2024.",
        "output": '{"people":[],"dates":["16 al 26 ottobre 2024"],"places":["Auditorium Parco della Musica"]}'
    },
    {
        "document": "Sergio Mattarella parteciperà alla cerimonia presso il Vittoriano a Roma il 2 giugno 2024.",
        "output": '{"people":["Sergio Mattarella"],"dates":["2 giugno 2024"],"places":["Vittoriano a Roma"]}'
    },
    {
        "document": "La conferenza di gastronomia molecolare con Ferran Adrià si terrà presso l'Università di Scienze Gastronomiche di Pollenzo.",
        "output": '{"people":["Ferran Adrià"],"dates":[],"places":["Università di Scienze Gastronomiche di Pollenzo"]}'
    },
    {
        "document": "Il convegno di neuroscienze avrà luogo presso l'Istituto San Raffaele di Milano il 12 settembre 2024.",
        "output": '{"people":[],"dates":["12 settembre 2024"],"places":["Istituto San Raffaele di Milano"]}'
    },
    {
        "document": "La mostra di Caravaggio sarà esposta presso i Musei Capitolini di Roma dal 5 marzo al 29 giugno 2024.",
        "output": '{"people":["Caravaggio"],"dates":["5 marzo al 29 giugno 2024"],"places":["Musei Capitolini di Roma"]}'
    },
    {
        "document": "Il Prof. Carlo Rubbia terrà una conferenza sulla fisica delle particelle presso il CERN di Ginevra.",
        "output": '{"people":["Carlo Rubbia"],"dates":[],"places":["CERN di Ginevra"]}'
    },
    {
        "document": "Il festival della canzone italiana si svolgerà presso il Teatro Ariston di Sanremo dal 7 all'11 febbraio 2024.",
        "output": '{"people":[],"dates":["7 all\'11 febbraio 2024"],"places":["Teatro Ariston di Sanremo"]}'
    },
    {
        "document": "La presentazione del libro di Umberto Galimberti si terrà presso la Biblioteca Marciana di Venezia.",
        "output": '{"people":["Umberto Galimberti"],"dates":[],"places":["Biblioteca Marciana di Venezia"]}'
    },
    {
        "document": "Il concorso di cucina vedrà la partecipazione di Gino Strada presso l'Expo Milano il 20 maggio 2024.",
        "output": '{"people":["Gino Strada"],"dates":["20 maggio 2024"],"places":["Expo Milano"]}'
    },
    {
        "document": "La conferenza sulla sostenibilità ambientale si terrà presso l'Università Statale di Milano il 5 dicembre 2024.",
        "output": '{"people":[],"dates":["5 dicembre 2024"],"places":["Università Statale di Milano"]}'
    },
    {
        "document": "Roberto Saviano presenterà il suo nuovo romanzo presso la Fiera del Libro di Torino.",
        "output": '{"people":["Roberto Saviano"],"dates":[],"places":["Fiera del Libro di Torino"]}'
    },
    {
        "document": "Il convegno di archeologia si svolgerà presso il Museo Archeologico Nazionale di Napoli dal 15 al 18 aprile 2024.",
        "output": '{"people":[],"dates":["15 al 18 aprile 2024"],"places":["Museo Archeologico Nazionale di Napoli"]}'
    },
    {
        "document": "La Dr.ssa Rita Levi Montalcini terrà una conferenza sulla ricerca scientifica presso l'Università di Pavia.",
        "output": '{"people":["Rita Levi Montalcini"],"dates":[],"places":["Università di Pavia"]}'
    },
    {
        "document": "Il festival della pizza napoletana si terrà in Piazza del Plebiscito a Napoli dal 10 al 15 settembre 2024.",
        "output": '{"people":[],"dates":["10 al 15 settembre 2024"],"places":["Piazza del Plebiscito a Napoli"]}'
    },
    {
        "document": "La conferenza di Piero Angela sulla divulgazione scientifica avrà luogo presso il Palazzo della Scienza di Milano.",
        "output": '{"people":["Piero Angela"],"dates":[],"places":["Palazzo della Scienza di Milano"]}'
    },
    {
        "document": "Il summit economico europeo si svolgerà presso Villa San Martino a Arcore dal 25 al 27 giugno 2024.",
        "output": '{"people":[],"dates":["25 al 27 giugno 2024"],"places":["Villa San Martino a Arcore"]}'
    },
    {
        "document": "La mostra di arte moderna si terrà presso la Galleria Nazionale d'Arte Moderna di Roma fino al 31 gennaio 2025.",
        "output": '{"people":[],"dates":["31 gennaio 2025"],"places":["Galleria Nazionale d\'Arte Moderna di Roma"]}'
    }
]

def main():
    """Create final expanded dataset."""
    # Read existing combined data
    train_examples = []
    val_examples = []
    
    # Load existing training data
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_combined.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    train_examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON in training data: {line[:100]}...")
    
    # Load existing validation data  
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_combined.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    val_examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON in validation data: {line[:100]}...")
    
    # Add new high-quality examples
    # Split the 50 additional examples: 40 for training, 10 for validation
    train_examples.extend(ADDITIONAL_EXAMPLES[:40])
    val_examples.extend(ADDITIONAL_EXAMPLES[40:])
    
    # Write final training dataset
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_final.jsonl', 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Write final validation dataset
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_final.jsonl', 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Final dataset created:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Total examples: {len(train_examples) + len(val_examples)}")

if __name__ == "__main__":
    main()