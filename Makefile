.PHONY: install test pretrain finetune evaluate demo shapes lint

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

shapes:
	python scripts/inspect_shapes.py

prepare_data:
	python scripts/prepare_ubfc.py --root $(MASKFUSIONNET_UBFC_ROOT)

pretrain:
	python scripts/train_pretrain.py --config configs/pretrain.yaml

finetune:
	python scripts/train_finetune.py --config configs/finetune.yaml

evaluate:
	python scripts/evaluate.py --config configs/finetune.yaml --checkpoint checkpoints/finetune/best.pt

demo:
	python scripts/realtime_demo.py --checkpoint checkpoints/finetune/best.pt
