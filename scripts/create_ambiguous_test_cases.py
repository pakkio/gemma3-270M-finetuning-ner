#!/usr/bin/env python3
"""
Crea test cases ambigui per nomi italiani per test più robusti.
"""

import json
from pathlib import Path

def create_ambiguous_test_cases():
    """Crea test cases che sfidano davvero il modello."""
    
    # Test cases ambigui reali
    ambiguous_cases = [
        {
            "document": "Romano Prodi ha incontrato il sindaco nella capitale dell'impero romano.",
            "expected": {"people": ["Romano Prodi"], "places": [], "dates": []},
            "challenge": "Romano = nome vs aggettivo"
        },
        {
            "document": "Il sindaco Marino ha visitato la città di Marino nel Lazio.",
            "expected": {"people": ["Marino"], "places": ["Marino", "Lazio"], "dates": []},
            "challenge": "Marino = persona vs luogo"
        },
        {
            "document": "La famiglia Sicilia ha origini nella regione Sicilia.",
            "expected": {"people": ["Sicilia"], "places": ["Sicilia"], "dates": []},
            "challenge": "Sicilia = cognome vs regione"
        },
        {
            "document": "San Francesco d'Assisi è nato ad Assisi nel 1181.",
            "expected": {"people": ["San Francesco d'Assisi"], "places": ["Assisi"], "dates": ["1181"]},
            "challenge": "San Francesco = santo storico"
        },
        {
            "document": "Andrea è andato a casa di Andrea Bocelli ieri sera.",
            "expected": {"people": ["Andrea", "Andrea Bocelli"], "places": [], "dates": ["ieri sera"]},
            "challenge": "Nomi duplicati in contesti diversi"
        },
        {
            "document": "Il Prof. De Sanctis dell'Università degli Studi di Milano-Bicocca ha parlato.",
            "expected": {"people": ["De Sanctis"], "places": ["Università degli Studi di Milano-Bicocca"], "dates": []},
            "challenge": "Entity compound e titoli"
        },
        {
            "document": "Marco, Maria e Giuseppe si sono incontrati a Piazza San Marco.",
            "expected": {"people": ["Marco", "Maria", "Giuseppe"], "places": ["Piazza San Marco"], "dates": []},
            "challenge": "Multiple entities, Marco = nome vs piazza"
        },
        {
            "document": "Il dott. Rossi ha lavorato dal 2020 al 2023 presso l'ospedale.",
            "expected": {"people": ["Rossi"], "places": [], "dates": ["dal 2020 al 2023"]},
            "challenge": "Titoli e periodi temporali"
        },
        {
            "document": "Via Roma, 15 - qui abita la signora Roma Paolini.",
            "expected": {"people": ["Roma Paolini"], "places": ["Via Roma"], "dates": []},
            "challenge": "Via vs nome persona"
        },
        {
            "document": "Durante l'era di Augusto, l'impero romano raggiunse la massima espansione.",
            "expected": {"people": ["Augusto"], "places": [], "dates": []},
            "challenge": "Personaggi storici vs aggettivi"
        }
    ]
    
    # Test cases con boundary detection difficili
    boundary_cases = [
        {
            "document": "Jean-Luc van der Berg ha visitato Saint-Pierre-et-Miquelon.",
            "expected": {"people": ["Jean-Luc van der Berg"], "places": ["Saint-Pierre-et-Miquelon"], "dates": []},
            "challenge": "Nomi composti con trattini e preposizioni"
        },
        {
            "document": "Maria Rossi-Bianchi e Anna De Santis si sono laureate.",
            "expected": {"people": ["Maria Rossi-Bianchi", "Anna De Santis"], "places": [], "dates": []},
            "challenge": "Cognomi doppi e particelle nobiliari"
        },
        {
            "document": "L'Ing. Carlo Da Vinci Jr. ha progettato il ponte.",
            "expected": {"people": ["Carlo Da Vinci Jr."], "places": [], "dates": []},
            "challenge": "Titoli, particelle, suffissi"
        }
    ]
    
    # Test cases con contesti colloquiali
    colloquial_cases = [
        {
            "document": "Ieri Luca ha detto a Giova di incontrare Ale al bar.",
            "expected": {"people": ["Luca", "Giova", "Ale"], "places": [], "dates": ["ieri"]},
            "challenge": "Nomi abbreviati colloquiali"
        },
        {
            "document": "Il boss della 'ndrangheta è stato arrestato a Reggio Calabria.",
            "expected": {"people": [], "places": ["Reggio Calabria"], "dates": []},
            "challenge": "Slang e riferimenti indiretti"
        }
    ]
    
    all_cases = ambiguous_cases + boundary_cases + colloquial_cases
    
    return all_cases

def save_test_cases():
    """Salva i test cases per evaluation."""
    cases = create_ambiguous_test_cases()
    
    output_dir = Path("data/challenging_tests")
    output_dir.mkdir(exist_ok=True)
    
    # Salva in formato JSONL per evaluation
    with open(output_dir / "ambiguous_test_cases.jsonl", 'w', encoding='utf-8') as f:
        for case in cases:
            jsonl_format = {
                "document": case["document"],
                "output": json.dumps(case["expected"], ensure_ascii=False)
            }
            f.write(json.dumps(jsonl_format, ensure_ascii=False) + '\n')
    
    # Salva con metadati per analisi
    with open(output_dir / "ambiguous_test_cases_detailed.json", 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(cases)} challenging test cases")
    print(f"Saved to: {output_dir}")
    
    return cases

def analyze_challenges():
    """Analizza i tipi di challenge nei test cases."""
    cases = create_ambiguous_test_cases()
    
    challenges = {}
    for case in cases:
        challenge_type = case["challenge"]
        if challenge_type not in challenges:
            challenges[challenge_type] = []
        challenges[challenge_type].append(case["document"])
    
    print("\n🔍 CHALLENGE CATEGORIES:")
    for challenge, examples in challenges.items():
        print(f"\n{challenge}:")
        for example in examples:
            print(f"  - {example}")
    
    return challenges

if __name__ == "__main__":
    print("🧪 CREATING CHALLENGING TEST CASES FOR ROBUST EVALUATION")
    print("=" * 60)
    
    cases = save_test_cases()
    analyze_challenges()
    
    print(f"\n✅ Created {len(cases)} challenging test cases")
    print("These cases will reveal if People F1=1.0 is real or overfitting!")