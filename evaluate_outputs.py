#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import unicodedata
from collections import Counter
from rapidfuzz import fuzz

def normalize(s):
    if not isinstance(s, str):
        return s
    s = s.lower().strip()
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')  # remove accents
    s = s.replace("’", "'").replace('“','"').replace('”','"')
    s = s.replace('.', '').replace(',', '').replace('–', '-')
    s = s.replace('  ', ' ')
    # Expand common Italian abbreviations
    s = s.replace('v.le', 'viale').replace('viale', 'viale')
    s = s.replace('p.zza', 'piazza').replace('piazza', 'piazza')
    s = s.replace('c.so', 'corso').replace('c.so', 'corso')
    s = s.replace('s.da', 'strada').replace('s.da', 'strada')
    s = s.replace('s.s.', 'strada statale').replace('s.s', 'strada statale')
    s = s.replace('s.p.', 'strada provinciale').replace('s.p', 'strada provinciale')
    s = s.replace('s.r.', 'strada regionale').replace('s.r', 'strada regionale')
    s = s.replace('citta', 'città')
    return s

def setify(lst):
    return set([normalize(x) for x in lst if x and isinstance(x, str)])

def compute_metrics(gold, pred, threshold=85):
    gold_norm = [normalize(x) for x in gold if x and isinstance(x, str)]
    pred_norm = [normalize(x) for x in pred if x and isinstance(x, str)]
    matched_gold = set()
    matched_pred = set()
    for i, g in enumerate(gold_norm):
        for j, p in enumerate(pred_norm):
            score = fuzz.token_set_ratio(g, p)
            if score >= threshold:
                matched_gold.add(i)
                matched_pred.add(j)
                break
    tp = len(matched_gold)
    fp = len(pred_norm) - len(matched_pred)
    fn = len(gold_norm) - len(matched_gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

def evaluate(gold_outputs, pred_outputs):
    categories = ['people', 'dates', 'places']
    results = {cat: {'tp':0, 'fp':0, 'fn':0, 'precision_sum':0.0, 'recall_sum':0.0, 'f1_sum':0.0} for cat in categories}
    total_tp = total_fp = total_fn = 0
    n = len(gold_outputs)
    for gold, pred in zip(gold_outputs, pred_outputs):
        for cat in categories:
            m = compute_metrics(gold.get(cat, []), pred.get(cat, []))
            for k in ['tp', 'fp', 'fn']:
                results[cat][k] += m[k]
            results[cat]['precision_sum'] += m['precision']
            results[cat]['recall_sum'] += m['recall']
            results[cat]['f1_sum'] += m['f1']
        total_tp += sum([compute_metrics(gold.get(cat, []), pred.get(cat, []))['tp'] for cat in categories])
        total_fp += sum([compute_metrics(gold.get(cat, []), pred.get(cat, []))['fp'] for cat in categories])
        total_fn += sum([compute_metrics(gold.get(cat, []), pred.get(cat, []))['fn'] for cat in categories])
    print("\n=== RISULTATI VALUTAZIONE FUZZY ===")
    for cat in categories:
        tp = results[cat]['tp']
        fp = results[cat]['fp']
        fn = results[cat]['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\nCategoria: {cat}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1:        {f1:.2f}")
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    print(f"\nMICRO-AVERAGE (global):\n  Precision: {micro_precision:.2f}\n  Recall:    {micro_recall:.2f}\n  F1:        {micro_f1:.2f}")

def main():
    # Inserisci qui i gold standard e i risultati del modello
    gold_outputs = [
        {"people": ["Maria De Santis", "Luca Bianchi"], "dates": ["15 agosto 2025", "15 settembre 2025"], "places": ["Roma", "Museo di Palazzo Barberini"]},
        {"people": ["Marta Verdi"], "dates": ["5 giugno 2024"], "places": ["Milano", "Politecnico di Milano"]},
        {"people": ["Giovanni Allevi"], "dates": ["8 novembre 2024", "1° ottobre 2024"], "places": ["Torino", "Teatro Regio"]},
        {"people": ["Matteo Ricci"], "dates": ["14 febbraio 2025"], "places": ["Venezia", "Ca' Pesaro"]},
        {"people": ["Federico Neri"], "dates": ["22 dicembre 2024"], "places": ["Genova", "Università di Genova", "Aula 101"]},
        {"people": ["L'Angelo D'Amico", "Maria O'Connor", "Gennaro Dell'Isola"], "dates": ["3 marzo 2025"], "places": ["Roma", "Università di Tor Vergata", "Aula Magna", "Via della Ricerca Scientifica 1"]},
        {"people": [], "dates": ["primavera 2025", "fine aprile 2025", "inizio maggio 2025"], "places": ["Napoli", "Teatro San Carlo", "Piazza Trieste e Trento", "Galleria Umberto I"]},
        {"people": ["Jean-Pierre L'Écuyer", "Anna-Maria Müller-Schmidt", "Sam Altman"], "dates": ["12/06/2025"], "places": ["Firenze", "Université de Montréal", "ETH Zürich", "OpenAI", "Aula Magna", "UniFi", "Viale Morgagni 40"]},
        {"people": ["Clara Schumann", "Roberto Valle"], "dates": ["30 aprile 2024"], "places": ["Palermo", "Teatro Massimo"]},
        {"people": ["Carlo Rovelli"], "dates": ["25 luglio 2024", "28 luglio"], "places": ["Trieste", "Teatro Romano"]}
    ]
    pred_outputs = [
        {"people": ["Maria De Santis", "Luca Bianchi"], "dates": ["15 settembre 2025", "Museo di Palazzo Barberini", "Roma", "Italia"], "places": ["Roma", "Palazzo Barberini", "Museo di Palazzo Barberini"]},
        {"people": ["Milano", "5 giugno 2024"], "dates": ["5 giugno 2024"], "places": ["Milano", "5 giugno 2024"]},
        {"people": [], "dates": [], "places": []},
        {"people": [], "dates": [], "places": []},
        {"people": ["Federico Neri", "Federico Neri"], "dates": ["22 dicembre 2024", "15:00"], "places": []},
        {"people": ["Angelo D'Amico", "Maria O'Connor", "Gennaro Dell'Isola", "Sen. Gennaro Dell'Isola"], "dates": [], "places": []},
        {"people": ["Napoli", "Springa", "2025"], "dates": ["2025", "Primavera", "2025"], "places": ["San Carlo", "Trio di Trento", "Galleria Umberto I", "Teatro San Carlo", "Piazza Trieste e Trento", "GalleriamubertoI"]},
        {"people": ["Jean-Pierre L'Écuyer", "Anna-Maria Müller-Schmidt", "Sam Altman"], "dates": ["12/06/2025"], "places": ["V.le Morgagni 40", "UniFi", "Aula Magna", "V.le Morgagni 40"]},
        {"people": ["Clara Schumann"], "dates": ["30 aprile 2024"], "places": []},
        {"people": ["Carlo Rovelli"], "dates": ["25 luglio 2024", "13 luglio 2024"], "places": []}
    ]
    evaluate(gold_outputs, pred_outputs)

if __name__ == "__main__":
    main()
