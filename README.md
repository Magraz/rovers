### To Install

Run the following commands:
1. `pip install -r requirements.txt`
2. `sudo apt install build-essential`
3. `sudo apt install python3-dev`
4. `make clean && make release entry=rovers && ./build/bin/rovers`
5. `pip install -e .`

### To Run Experiment

Run the following command for running experiments with all default values:
- `python3 pyrover/run_experiment.py`

Run the following command for running experiments in different modalities:

- `python3 pyrover_domain/run_experiment.py --poi_type static --model mlp --teaming`

- `python3 pyrover_domain/run_experiment.py --poi_type decay --model mlp --teaming`

- `python3 pyrover_domain/run_experiment.py --poi_type decay --model gru --teaming`

- `python3 pyrover_domain/run_experiment.py --poi_type static --model mlp --teaming --hpc`
