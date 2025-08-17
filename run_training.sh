#!/bin/bash

# Script di avvio rapido per il fine-tuning con Poetry
# Modifica i parametri secondo le tue esigenze

CONFIG="gemma3_270m_balanced"  # fast, balanced, quality

echo "🚀 Starting fine-tuning with Poetry"
echo "Configuration: $CONFIG"
echo

# Verifica se Poetry è installato
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry non trovato. Installa con:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Verifica se le dipendenze sono installate
if [ ! -f "poetry.lock" ] || [ ! -d ".venv" ]; then
    echo "📦 Installing dependencies with Poetry..."
    poetry install
    echo
fi

# Avvia il training
echo "🔥 Starting training..."
poetry run run-config --config $CONFIG

# Verifica se il training è andato a buon fine
if [ $? -eq 0 ]; then
    echo
    echo "✅ Training completed!"
    echo "Model saved in: outputs/gemma3-270m-$CONFIG"
    echo
    echo "🧪 To test the model, run:"
    echo "poetry run inference --model_path outputs/gemma3-270m-$CONFIG --interactive"
    echo
    echo "📊 To evaluate the model, run:"
    echo "poetry run evaluate --model_path outputs/gemma3-270m-$CONFIG"
else
    echo "❌ Training failed!"
    exit 1
fi