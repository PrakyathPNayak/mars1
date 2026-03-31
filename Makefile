.PHONY: install demo train eval test dashboard clean explore record model

install:
	pip install gymnasium stable-baselines3[extra] tensorboard mujoco pynput imageio scipy matplotlib

model:
	python3 scripts/generate_model.py

demo:
	python3 run.py demo

demo-no-policy:
	python3 run.py demo --no-policy

train:
	python3 run.py train --steps 5000000 --envs 20

train-quick:
	python3 run.py train --steps 500000 --envs 4

eval:
	python3 run.py eval

test:
	python3 run.py test

dashboard:
	python3 run.py dashboard

explore:
	python3 run.py explore --heading 45

record:
	python3 run.py demo --record

clean:
	rm -rf logs/ __pycache__/ **/__pycache__/ .pytest_cache/

clean-all: clean
	rm -rf checkpoints/ research/papers/*.pdf
