BASE_OLLAMA ?= qwen3:4b-instruct-2507-q4_K_M
BASE_MLX    ?= mlx-community/Qwen3-4B-Instruct-2507-4bit

.PHONY: test prep baselines smoke eval-smoke train eval-adapter export eval-export ablation compare

test:
	pytest -q

prep:
	scrubber prep --out data

baselines:
	scrubber eval --backend regex --tag regex
	scrubber eval --backend presidio --tag presidio
	ollama pull $(BASE_OLLAMA)
	scrubber eval --backend ollama --model $(BASE_OLLAMA) --tag base_prompted

smoke:
	mlx_lm.lora --config configs/lora_smoke_1_7b.yaml

eval-smoke:
	scrubber eval --backend mlx --model mlx-community/Qwen3-1.7B-4bit --adapter adapters/smoke --limit 50 --tag smoke

train:
	mlx_lm.lora --config configs/lora_qwen3_4b.yaml

eval-adapter:
	scrubber eval --backend mlx --model $(BASE_MLX) --adapter adapters/qwen3-4b --tag finetuned_mlx

export:
	./scripts/export_ollama.sh adapters/qwen3-4b $(BASE_MLX) scrubber

eval-export:
	scrubber eval --backend ollama --model scrubber --tag finetuned_ollama

# Phase 3: does domain data help? Train on general data only and compare.
ablation:
	mlx_lm.lora --config configs/lora_qwen3_4b.yaml --data data/mlx_general_only --adapter-path adapters/general_only
	scrubber eval --backend mlx --model $(BASE_MLX) --adapter adapters/general_only --tag general_only

compare:
	scrubber compare results/*.json
