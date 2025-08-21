#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.inference import load_model_and_tokenizer, extract_entities
import json

test_examples = [
    # 1. Standard
    "Roma, 15 agosto 2025 — L'assessora alla cultura Maria De Santis ha presentato il nuovo programma museale insieme al direttore Luca Bianchi. L'evento si terrà al Museo di Palazzo Barberini il 15 settembre 2025.",
    # 2. Standard
    "Milano, 5 giugno 2024 — Conferenza di Marta Verdi al Politecnico di Milano sulla ricerca in intelligenza artificiale. L'evento ha visto la partecipazione di oltre 200 studenti.",
    # 3. Standard
    "Torino, 8 novembre 2024 — Il concerto di Giovanni Allevi si terrà al Teatro Regio. I biglietti sono disponibili dal 1° ottobre 2024.",
    # 4. Standard
    "Venezia, 14 febbraio 2025 — Inaugurazione della mostra 'Arte Contemporanea' presso Ca' Pesaro. Il curatore Matteo Ricci ha selezionato opere di 50 artisti internazionali.",
    # 5. Standard
    "Genova, 22 dicembre 2024 — Il dottor Federico Neri presenterà la sua ricerca sul cambiamento climatico all'Università di Genova. L'incontro è previsto per le ore 15:00 nell'Aula 101.",
    # 6. Edge-case: nomi con apostrofi, titoli, errori
    "Roma, 3 marzo 2025 — Il Prof. Dott. L'Angelo D'Amico, insieme alla Dott.ssa Maria O'Connor e al Sen. Gennaro Dell'Isola, ha presentato il progetto 'Nuove frontiere dell'IA' presso l'Univ. di Tor Vergata, Aula Magna, Via della Ricerca Scentifica 1 (errore voluto).",
    # 7. Edge-case: date vaghe, luoghi con errori
    "Napoli, primavera 2025 — Festival 'Musica e Parole' con inizio previsto tra fine aprile e inizio magio (errore voluto). Eventi principali al Teatro San Carlo, Piazza Trieste e Trento, e presso la Galleria Umberto I.",
    # 8. Edge-case: entità annidate, acronimi, nomi stranieri
    "Firenze, 12/06/2025 — Conferenza internazionale 'AI & Society' con keynote di Dr. Jean-Pierre L'Écuyer (Université de Montréal), Prof.ssa Anna-Maria Müller-Schmidt (ETH Zürich), e CEO di OpenAI Sam Altman. Sede: Aula Magna, UniFi, V.le Morgagni 40.",
    # 9. Standard
    "Palermo, 30 aprile 2024 — Il festival di musica classica vedrà esibirsi la pianista Clara Schumann al Teatro Massimo. L'evento è organizzato dal Maestro Roberto Valle.",
    # 10. Standard
    "Trieste, 25 luglio 2024 — Il festival della scienza ospiterà il fisico Carlo Rovelli presso il Teatro Romano. La conferenza 'Il tempo e lo spazio' è prevista per domenica 28 luglio."
]

def main():
    print("\U0001F9EA Testing nuovo modello edgecase con 10 esempi...")
    print("=" * 80)
    model_path = "outputs/gemma3-270m-edgecase"
    model, tokenizer = load_model_and_tokenizer(model_path)
    print(f"✅ Modello caricato da: {model_path}")
    print("=" * 80)

    for i, document in enumerate(test_examples, 1):
        print(f"\n🔍 TEST {i}/10:")
        print("📄 Documento:")
        print(document[:300] + "..." if len(document) > 300 else document)
        print("\n🎯 Risultato:")
        try:
            result, raw_output = extract_entities(model, tokenizer, document)
            if result:
                print("✅ JSON valido estratto:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                total_entities = len(result.get('people', [])) + len(result.get('dates', [])) + len(result.get('places', []))
                print(f"📊 Entità totali: {total_entities} (👥 {len(result.get('people', []))} persone, 📅 {len(result.get('dates', []))} date, 📍 {len(result.get('places', []))} luoghi)")
            else:
                print("❌ Errore nell'estrazione")
                print(f"Raw output: {raw_output[:100]}...")
        except Exception as e:
            print(f"❌ Errore: {e}")
        print("-" * 60)
    print("\n🎉 Test completato!")

if __name__ == "__main__":
    main()
