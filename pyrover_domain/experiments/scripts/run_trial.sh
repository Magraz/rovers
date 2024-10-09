TRIAL_ID=${1}
python3 /home/magraz/rovers/pyrover_domain/run_experiment.py --experiment_type fitness_critic  --poi_type static --model gru --trial_id ${TRIAL_ID}
echo "Finished Trial #${TRIAL_ID}"