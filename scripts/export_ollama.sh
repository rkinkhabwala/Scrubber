#!/usr/bin/env bash
# Fuse the LoRA adapter, convert to GGUF and register the model with Ollama as "scrubber".
#
#   ./scripts/export_ollama.sh [adapter_dir] [base_model] [ollama_name]
#
# Needs: mlx-lm, a llama.cpp checkout (for convert_hf_to_gguf.py) and Ollama running.
set -euo pipefail

ADAPTER=${1:-adapters/qwen3-4b}
BASE=${2:-mlx-community/Qwen3-4B-Instruct-2507-4bit}
NAME=${3:-scrubber}
LLAMA_CPP=${LLAMA_CPP:-$HOME/src/llama.cpp}
OUT=build/$NAME

mkdir -p "$OUT"

echo "==> 1/4 Fusing adapter into base model (dequantized to 16-bit for conversion)"
mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTER" --save-path "$OUT/fused" --dequantize

echo "==> 2/4 Converting to GGUF (f16)"
if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
  echo "    llama.cpp not found at $LLAMA_CPP — cloning it (one time)"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
  pip install -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"
fi
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$OUT/fused" --outtype f16 --outfile "$OUT/$NAME-f16.gguf"

echo "==> 3/4 Writing Modelfile (system prompt baked in, greedy decoding)"
python - "$OUT" "$NAME" <<'PY'
import sys, pathlib
from scrubber.prompt import SYSTEM_PROMPT
out, name = pathlib.Path(sys.argv[1]), sys.argv[2]
tmpl = pathlib.Path("scripts/Modelfile.template").read_text()
(out / "Modelfile").write_text(
    tmpl.replace("{{GGUF}}", f"./{name}-f16.gguf").replace("{{SYSTEM}}", SYSTEM_PROMPT.replace('"""', "'''"))
)
PY

echo "==> 4/4 Creating Ollama model '$NAME' (quantized to q4_K_M)"
(cd "$OUT" && ollama create "$NAME" -q q4_K_M -f Modelfile)

echo
echo "Done. Try:  scrubber redact \"Hi, I'm Maria Lopez, card 4111 1111 1111 1111\" --model $NAME"
echo "Evaluate:   scrubber eval --backend ollama --model $NAME --tag finetuned_ollama"
echo "You can delete $OUT/fused and the .gguf afterwards to free disk space."
