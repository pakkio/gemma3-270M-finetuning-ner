#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.inference import load_model_and_tokenizer, extract_entities
import json

test_edge_cases = [
    # 1. Nomi con apostrofi, errori di battitura e titoli misti
    """Roma, 3 marzo 2025 — Il Prof. Dott. L'Angelo D'Amico, insieme alla Dott.ssa Maria O'Connor e al Sen. Gennaro Dell'Isola, ha presentato il progetto \"Nuove frontiere dell'IA\" presso l'Univ. di Tor Vergata, Aula Magna, Via della Ricerca Scentifica 1 (errore voluto).""",

    # 2. Date incomplete, periodi vaghi, luoghi con errori
    """Napoli, primavera 2025 — Festival \"Musica e Parole\" con inizio previsto tra fine aprile e inizio magio (errore voluto). Eventi principali al Teatro San Carlo, Piazza Trieste e Trento, e presso la Galleria Umberto I.""",

    # 3. Entità annidate, acronimi, nomi stranieri e formati misti
    """Firenze, 12/06/2025 — Conferenza internazionale \"AI & Society\" con keynote di Dr. Jean-Pierre L'Écuyer (Université de Montréal), Prof.ssa Anna-Maria Müller-Schmidt (ETH Zürich), e CEO di OpenAI Sam Altman. Sede: Aula Magna, UniFi, V.le Morgagni 40.""",

    # 4. Luoghi composti, date relative, nomi con caratteri speciali
    """Palermo, tra il 10 e il 15 luglio 2025 — Workshop \"Innovazione & Territorio\" con la partecipazione di Dr. François D'Haene, Prof. María-José García López e l'arch. Luigi D'Angelo. Location: Palazzo dei Normanni, Sala dei Viceré, Piazza Indipendenza.""",

    # 5. Testo multilingue, date in formato testuale, entità miste
    """Bolzano, 1st September 2025 — Das Event \"Grenzenlos Forschen\" findet am NOI Techpark statt. Ospiti: Dr. Hans-Jürgen Weber, Prof.ssa Elena Kozlova, e il Sindaco Renzo Caramaschi. Termine: primo lunedì di settembre."""
]

def main():
    print("\U0001F9EA Testing modello con 5 edge-case difficili...")
    print("=" * 80)
    model_path = "outputs/gemma3-270m-quality-expanded"
    model, tokenizer = load_model_and_tokenizer(model_path)
    print(f"✅ Modello caricato da: {model_path}")
    print("=" * 80)

    for i, document in enumerate(test_edge_cases, 1):
        print(f"\n🔍 EDGE TEST {i}/5:")
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
    print("\n🎉 Edge-case test completato!")

if __name__ == "__main__":
    main()
