#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.inference import load_model_and_tokenizer, extract_entities
import json

# Test del modello migliorato con 10 esempi concreti
test_examples = [
    # 1. Documento complesso originale
    """Comunicato stampa — Rete Musei Civici Roma, 15 agosto 2025 — L'assessora alla cultura Maria De Santis ha presentato oggi, insieme al curatore Luca Bianchi, la mostra "Cartografie del Tempo" presso il Museo di Palazzo Barberini. Il progetto, coordinato da Giulia Rossi in collaborazione con l'Università di Bologna, sarà aperto al pubblico dal 15/09/2025 al 17/11/2025, con anteprima stampa il 02.01.23 (data storica dell'accordo preliminare tra gli enti). Durante l'evento, l'ospite internazionale Ana-María López ha tenuto una breve lectio sulla relazione tra memoria e territorio, con riferimenti a Palermo e al Lago di Como. Una tappa itinerante è prevista a Milano (Via del Corso 12, sede distaccata) dal 15–17 giugno 2024, quindi trasferimento al Teatro Massimo, Palermo per un concerto conclusivo il 01/12/2025. "L'obiettivo," ha dichiarato Dott. Federico Neri, "è connettere archivi e percorsi urbani." Per informazioni: Ufficio Stampa, Piazza Navona, Roma.""",
    
    # 2. Nomi con titoli complessi
    """Milano, 14 novembre 2024 — Conferenza 'Fintech e Sostenibilità' con il Rettore Prof. Gianmario Verona, CEO Intesa Sanpaolo Dott. Carlo Messina e fintech expert Prof.ssa Paola Schwizer presso l'Università Bocconi, Aula Magna Leone, Via Bocconi 8.""",
    
    # 3. Date in formati vari
    """Bari, dal 15 al 20 maggio 2024 — Congresso internazionale di neurologia. Tra i relatori: Dr. Michael Johnson (Harvard), Prof.ssa Elena Kozlova (Mosca), Dr. Jean-Luc Moreau (Sorbonne). Evento dal lunedì alla domenica, inaugurazione il 15/05/2024 ore 9:00.""",
    
    # 4. Luoghi internazionali e indirizzi
    """Trieste, ogni mercoledì di gennaio 2025 — Seminario di fisica teorica con Nobel Prof. Giorgio Parisi (Sapienza Roma), Dr. Guido Martinelli (SISSA) e ricercatrice Dr.ssa Anna Ceresole (INFN-Torino). Edificio H3, Via Bonomea 265, Trieste.""",
    
    # 5. Nomi stranieri complessi
    """Bolzano/Bozen, 30.11.2024 — Der Landeshauptmann Dr. Arno Kompatscher und Dr. Martha Müller werden das neue Forschungszentrum eröffnen. L'evento bilingue si terrà presso il NOI Techpark in Via A. Volta 13/A. Ospite d'onore: Prof. Hans-Jürgen Weber (Università di Vienna).""",
    
    # 6. Date relative e periodi
    """Cosenza, estate 2024 — Festival letterario 'Calabria Legge' ospiterà Roberto Benigni, Dacia Maraini e scrittore Francesco De Gregori (omonimo del cantautore). Eventi dal 20/06/2024 al 25/06/2024, con anteprima nel primo weekend di giugno.""",
    
    # 7. Molte persone e istituzioni
    """Venezia, durante la 81ª Mostra del Cinema (28 agosto - 7 settembre 2024) — Anteprima mondiale del film 'Corpo Celeste' con regista Alice Rohrwacher, attori Tilda Swinton e Josh O'Connor, produttore Mario Gianani. Presentazione al Palazzo del Cinema del Lido di Venezia con critica cinematografica Elena Pollacchi.""",
    
    # 8. Indirizzi e coordinate specifiche
    """Reggio Calabria, 8 aprile 2024 ore 10:30 — Inaugurazione del nuovo porto turistico con Sindaco Giuseppe Falcomatà, Comandante Guardia Costiera Cap. Salvatore Minella e Amm. Marina De Luca. Località: Porto Bolaro, Via Marina Bassa n. 45.""",
    
    # 9. Date con orari e dettagli temporali
    """Assisi, Basilica di San Francesco - 4 ottobre 2024 (festa patronale) — Celebrazione presieduta dal Card. Mauro Gambetti con partecipazione Presidente Mattarella e Ministro Cultura Alessandro Giuli. Concerto serale Maestro Giovanni Allevi. Piazza Inferiore dalle 10:00 alle 18:00.""",
    
    # 10. Caso complesso con acronimi e università
    """Como, Villa Olmo - 5-7 luglio 2024 — Workshop internazionale fisica particelle organizzato dal CERN con Prof. Fabiola Gianotti (Direttore Generale), fisico teorico Prof. Giorgio Parisi (Nobel) e ricercatrice Dr.ssa Elena Aprile (Columbia University). Giardini Villa Olmo, Lungolago Mafalda di Savoia."""
]

def main():
    print("🧪 Testing modello migliorato con 10 esempi concreti...")
    print("=" * 80)
    
    # Carica il modello migliorato
    model_path = "outputs/gemma3-270m-quality-expanded"
    model, tokenizer = load_model_and_tokenizer(model_path)
    print(f"✅ Modello caricato da: {model_path}")
    print("=" * 80)
    
    for i, document in enumerate(test_examples, 1):
        print(f"\n🔍 TEST {i}/10:")
        print("📄 Documento:")
        print(document[:200] + "..." if len(document) > 200 else document)
        print("\n🎯 Risultato:")
        
        try:
            result, raw_output = extract_entities(model, tokenizer, document)
            
            if result:
                print("✅ JSON valido estratto:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                # Conta entità estratte
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