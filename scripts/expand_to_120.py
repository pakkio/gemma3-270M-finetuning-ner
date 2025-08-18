#!/usr/bin/env python3
"""
Expand dataset to 120+ examples by adding more high-quality Italian examples.
"""

import json

# Additional 40 high-quality examples to reach 120+ total
MORE_EXAMPLES = [
    {
        "document": "Il sindaco di Roma Roberto Gualtieri inaugurerà la nuova metro presso la stazione Colosseo il 12 aprile 2024.",
        "output": '{"people":["Roberto Gualtieri"],"dates":["12 aprile 2024"],"places":["stazione Colosseo"]}'
    },
    {
        "document": "La conferenza di Matteo Renzi sulla politica europea si terrà presso l'Università LUISS di Roma.",
        "output": '{"people":["Matteo Renzi"],"dates":[],"places":["Università LUISS di Roma"]}'
    },
    {
        "document": "Il festival del gelato artigianale si svolgerà in Piazza Navona a Roma dal 1 al 10 agosto 2024.",
        "output": '{"people":[],"dates":["1 al 10 agosto 2024"],"places":["Piazza Navona a Roma"]}'
    },
    {
        "document": "La presentazione del libro di Paolo Cognetti avrà luogo presso la Biblioteca Nazionale di Firenze.",
        "output": '{"people":["Paolo Cognetti"],"dates":[],"places":["Biblioteca Nazionale di Firenze"]}'
    },
    {
        "document": "Il convegno di cardiologia si terrà presso l'Ospedale Gemelli di Roma dal 15 al 17 marzo 2024.",
        "output": '{"people":[],"dates":["15 al 17 marzo 2024"],"places":["Ospedale Gemelli di Roma"]}'
    },
    {
        "document": "Massimo Giletti condurrà il dibattito televisivo presso gli studi Mediaset di Milano.",
        "output": '{"people":["Massimo Giletti"],"dates":[],"places":["studi Mediaset di Milano"]}'
    },
    {
        "document": "La mostra di Van Gogh sarà esposta presso Palazzo Ducale di Mantova fino al 30 giugno 2024.",
        "output": '{"people":["Van Gogh"],"dates":["30 giugno 2024"],"places":["Palazzo Ducale di Mantova"]}'
    },
    {
        "document": "Il Prof. Giulio Giorello terrà una conferenza di filosofia della scienza presso l'Università Statale di Milano.",
        "output": '{"people":["Giulio Giorello"],"dates":[],"places":["Università Statale di Milano"]}'
    },
    {
        "document": "La sagra dell\'uva si svolgerà nel centro storico di Montalcino dal 20 al 25 settembre 2024.",
        "output": '{"people":[],"dates":["20 al 25 settembre 2024"],"places":["centro storico di Montalcino"]}'
    },
    {
        "document": "Gianni Morandi si esibirà in concerto presso il PalaDozza di Bologna il 18 novembre 2024.",
        "output": '{"people":["Gianni Morandi"],"dates":["18 novembre 2024"],"places":["PalaDozza di Bologna"]}'
    },
    {
        "document": "Il summit sulla sicurezza informatica avrà luogo presso il Centro Congressi Fiera Milano.",
        "output": '{"people":[],"dates":[],"places":["Centro Congressi Fiera Milano"]}'
    },
    {
        "document": "La Dr.ssa Ilaria Capua terrà una conferenza sui virus presso l\'Istituto Superiore di Sanità il 8 maggio 2024.",
        "output": '{"people":["Ilaria Capua"],"dates":["8 maggio 2024"],"places":["Istituto Superiore di Sanità"]}'
    },
    {
        "document": "Il festival del documentario si terrà presso il Cinema Odeon di Firenze dal 3 al 8 ottobre 2024.",
        "output": '{"people":[],"dates":["3 al 8 ottobre 2024"],"places":["Cinema Odeon di Firenze"]}'
    },
    {
        "document": "Claudio Amendola presenterà il suo film presso la Mostra del Cinema di Venezia.",
        "output": '{"people":["Claudio Amendola"],"dates":[],"places":["Mostra del Cinema di Venezia"]}'
    },
    {
        "document": "La conferenza di neurochirurgia vedrà la partecipazione del Prof. Giulio Maira presso l\'Ospedale San Camillo di Roma.",
        "output": '{"people":["Giulio Maira"],"dates":[],"places":["Ospedale San Camillo di Roma"]}'
    },
    {
        "document": "Il workshop di scrittura creativa si svolgerà presso la Casa delle Letterature di Roma dal 12 al 16 febbraio 2024.",
        "output": '{"people":[],"dates":["12 al 16 febbraio 2024"],"places":["Casa delle Letterature di Roma"]}'
    },
    {
        "document": "Maria De Filippi registrerà la puntata speciale negli studi Elios di Roma il 25 gennaio 2024.",
        "output": '{"people":["Maria De Filippi"],"dates":["25 gennaio 2024"],"places":["studi Elios di Roma"]}'
    },
    {
        "document": "Il convegno di sostenibilità ambientale avrà luogo presso l\'ENEA di Frascati.",
        "output": '{"people":[],"dates":[],"places":["ENEA di Frascati"]}'
    },
    {
        "document": "La mostra di Botticelli sarà inaugurata presso la Pinacoteca di Brera a Milano il 15 aprile 2024.",
        "output": '{"people":["Botticelli"],"dates":["15 aprile 2024"],"places":["Pinacoteca di Brera a Milano"]}'
    },
    {
        "document": "Il Prof. Stefano Mancuso terrà una conferenza sulle piante intelligenti presso l\'Università di Firenze.",
        "output": '{"people":["Stefano Mancuso"],"dates":[],"places":["Università di Firenze"]}'
    },
    {
        "document": "La gara ciclistica Giro d\'Italia partirà da Piazza del Duomo a Milano il 4 maggio 2024.",
        "output": '{"people":[],"dates":["4 maggio 2024"],"places":["Piazza del Duomo a Milano"]}'
    },
    {
        "document": "Alessandro Gassman presenterà il suo spettacolo teatrale presso il Teatro Eliseo di Roma.",
        "output": '{"people":["Alessandro Gassman"],"dates":[],"places":["Teatro Eliseo di Roma"]}'
    },
    {
        "document": "Il convegno di oncologia pediatrica si terrà presso l\'Ospedale Bambino Gesù di Roma dal 20 al 22 giugno 2024.",
        "output": '{"people":[],"dates":["20 al 22 giugno 2024"],"places":["Ospedale Bambino Gesù di Roma"]}'
    },
    {
        "document": "La chef Cristina Bowerman aprirà il nuovo locale presso Trastevere a Roma il 30 settembre 2024.",
        "output": '{"people":["Cristina Bowerman"],"dates":["30 settembre 2024"],"places":["Trastevere a Roma"]}'
    },
    {
        "document": "Il festival della musica classica si svolgerà presso Villa Adriana a Tivoli.",
        "output": '{"people":[],"dates":[],"places":["Villa Adriana a Tivoli"]}'
    },
    {
        "document": "Il Prof. Carlo Cottarelli terrà una lezione di economia presso l\'Università Cattolica di Milano il 10 marzo 2024.",
        "output": '{"people":["Carlo Cottarelli"],"dates":["10 marzo 2024"],"places":["Università Cattolica di Milano"]}'
    },
    {
        "document": "La mostra di fotografia di Helmut Newton sarà esposta presso il MUDEC di Milano fino al 28 luglio 2024.",
        "output": '{"people":["Helmut Newton"],"dates":["28 luglio 2024"],"places":["MUDEC di Milano"]}'
    },
    {
        "document": "Luciano Ligabue terrà un concerto unplugged presso il Teatro Comunale di Bologna il 14 maggio 2024.",
        "output": '{"people":["Luciano Ligabue"],"dates":["14 maggio 2024"],"places":["Teatro Comunale di Bologna"]}'
    },
    {
        "document": "Il summit internazionale sul turismo avrà luogo presso la Borsa Italiana di Milano.",
        "output": '{"people":[],"dates":[],"places":["Borsa Italiana di Milano"]}'
    },
    {
        "document": "La Dr.ssa Samantha Cristoforetti terrà una conferenza sullo spazio presso il Planetario di Milano il 21 giugno 2024.",
        "output": '{"people":["Samantha Cristoforetti"],"dates":["21 giugno 2024"],"places":["Planetario di Milano"]}'
    },
    {
        "document": "Il festival del teatro contemporaneo si terrà presso il Teatro Argentina di Roma dal 5 al 15 novembre 2024.",
        "output": '{"people":[],"dates":["5 al 15 novembre 2024"],"places":["Teatro Argentina di Roma"]}'
    },
    {
        "document": "Fabio Fazio condurrà lo speciale televisivo negli studi RAI di Torino.",
        "output": '{"people":["Fabio Fazio"],"dates":[],"places":["studi RAI di Torino"]}'
    },
    {
        "document": "La conferenza di intelligenza artificiale vedrà la partecipazione del Prof. Luciano Floridi presso il Politecnico di Milano.",
        "output": '{"people":["Luciano Floridi"],"dates":[],"places":["Politecnico di Milano"]}'
    },
    {
        "document": "Il mercato dell\'antiquariato si svolgerà in Piazza Grande ad Arezzo ogni prima domenica del mese.",
        "output": '{"people":[],"dates":["ogni prima domenica del mese"],"places":["Piazza Grande ad Arezzo"]}'
    },
    {
        "document": "Marco Mengoni si esibirà presso il Forum di Assago a Milano il 2 dicembre 2024.",
        "output": '{"people":["Marco Mengoni"],"dates":["2 dicembre 2024"],"places":["Forum di Assago a Milano"]}'
    },
    {
        "document": "Il convegno di medicina sportiva avrà luogo presso il CONI di Roma dal 8 al 10 aprile 2024.",
        "output": '{"people":[],"dates":["8 al 10 aprile 2024"],"places":["CONI di Roma"]}'
    },
    {
        "document": "La presentazione del vino Brunello si terrà presso il Castello Banfi di Montalcino il 19 ottobre 2024.",
        "output": '{"people":[],"dates":["19 ottobre 2024"],"places":["Castello Banfi di Montalcino"]}'
    },
    {
        "document": "Il Prof. Massimo Cacciari terrà una conferenza di filosofia presso Ca\' Foscari a Venezia.",
        "output": '{"people":["Massimo Cacciari"],"dates":[],"places":["Ca\' Foscari a Venezia"]}'
    },
    {
        "document": "La regata storica si svolgerà lungo il Canal Grande di Venezia il 1 settembre 2024.",
        "output": '{"people":[],"dates":["1 settembre 2024"],"places":["Canal Grande di Venezia"]}'
    },
    {
        "document": "Checco Zalone presenterà il suo nuovo film presso il Cinema Barberini di Roma il 20 dicembre 2024.",
        "output": '{"people":["Checco Zalone"],"dates":["20 dicembre 2024"],"places":["Cinema Barberini di Roma"]}'
    }
]

def main():
    """Add more examples to reach 120+ total."""
    # Load existing final data
    train_examples = []
    val_examples = []
    
    # Load existing training data
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_final.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_examples.append(json.loads(line))
    
    # Load existing validation data  
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_final.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                val_examples.append(json.loads(line))
    
    # Add more examples: 30 for training, 10 for validation  
    train_examples.extend(MORE_EXAMPLES[:30])
    val_examples.extend(MORE_EXAMPLES[30:])
    
    # Write final expanded datasets
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/train_expanded_final.jsonl', 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open('/home/pakkio/IdeaProjects/fine-tuning/data/val_expanded_final.jsonl', 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Expanded dataset created:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Total examples: {len(train_examples) + len(val_examples)}")

if __name__ == "__main__":
    main()