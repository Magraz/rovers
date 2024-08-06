from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm
import torch

import torch.multiprocessing as mp
import random

from pyrover_domain.models.mlp import MLP_Policy
from pyrover_domain.models.gru import GRU_Policy

from pyrover_domain.librovers import rovers
from pyrover_domain.custom_env import createEnv
from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JointTrajectory:
    def __init__(
        self,
        joint_state_trajectory,
        joint_observation_trajectory,
        joint_action_trajectory,
    ):
        self.states = joint_state_trajectory
        self.observations = joint_observation_trajectory
        self.actions = joint_action_trajectory


class EvalInfo:
    def __init__(self, fitnesses, joint_trajectory):
        self.fitnesses = fitnesses
        self.joint_trajectory = joint_trajectory


class CooperativeCoevolutionaryAlgorithm:
    def __init__(self, config_dir):
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = self.config_dir.parent

        with open(str(self.config_dir), "r") as file:
            self.config = yaml.safe_load(file)

        # Start by setting up variables for different agents
        self.num_rovers = len(self.config["env"]["agents"]["rovers"])
        self.subpopulation_size = self.config["ccea"]["population"]["subpopulation_size"]
        self.num_hidden = self.config["ccea"]["model"]["hidden_layers"]
        self.model_type = self.config["ccea"]["model"]["type"]

        self.num_pois = len(self.config["env"]["pois"])

        self.n_elites = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["n_elites"]
        self.include_elites_in_tournament = self.config["ccea"]["selection"]["n_elites_binary_tournament"][
            "include_elites_in_tournament"
        ]
        self.num_mutants = self.subpopulation_size - self.n_elites
        self.num_evaluations_per_team = self.config["ccea"]["evaluation"]["multi_evaluation"]["num_evaluations"]
        self.aggregation_method = self.config["ccea"]["evaluation"]["multi_evaluation"]["aggregation_method"]
        self.fitness_method = self.config["ccea"]["evaluation"]["fitness_method"]

        self.num_steps = self.config["ccea"]["num_steps"]

        if self.num_rovers > 0:
            self.num_rover_sectors = int(360 / self.config["env"]["agents"]["rovers"][0]["resolution"])
            self.rover_nn_template = self.generateTemplateNN(self.num_rover_sectors)
            self.rover_nn_size = self.rover_nn_template.num_params

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data saving variables
        self.save_trajectories = self.config["data"]["save_trajectories"]["switch"]
        self.num_gens_between_save_traj = self.config["data"]["save_trajectories"]["num_gens_between_save"]

        # Create the type of fitness we're optimizing
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()
        if self.use_multiprocessing:
            self.pool = mp.Pool(processes=self.num_threads)
            self.map = self.pool.map_async
        else:
            self.toolbox.register("map", map)
            self.map = map

    # This makes it possible to pass evaluation to multiprocessing
    # Without this, the pool tries to pickle the entire object, including itself
    # which it cannot do
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        del self_dict["map"]
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def generateTemplateNN(self, num_sectors):
        match (self.model_type):
            case "GRU":
                agent_nn = GRU_Policy(
                    input_size=2 * num_sectors,
                    hidden_size=self.num_hidden[0],
                    output_size=2,
                    num_layers=1,
                ).to(DEVICE)
            case "MLP":
                agent_nn = MLP_Policy(
                    input_size=2 * num_sectors,
                    hidden_size=self.num_hidden[0],
                    output_size=2,
                ).to(DEVICE)

        return agent_nn

    def generateWeight(self):
        return random.uniform(
            self.config["ccea"]["weight_initialization"]["lower_bound"],
            self.config["ccea"]["weight_initialization"]["upper_bound"],
        )

    def generateIndividual(self, individual_size):
        return tools.initRepeat(creator.Individual, self.generateWeight, n=individual_size)

    def generateRoverIndividual(self):
        return self.generateIndividual(individual_size=self.rover_nn_size)

    def generateRoverSubpopulation(self):
        return tools.initRepeat(
            list,
            self.generateRoverIndividual,
            n=self.config["ccea"]["population"]["subpopulation_size"],
        )

    def population(self):
        return tools.initRepeat(list, self.generateRoverSubpopulation, n=self.num_rovers)

    def formEvaluationTeam(self, population):
        eval_team = []
        for subpop in population:
            # Use max with a key function to get the individual with the highest fitness[0] value
            best_ind = max(subpop, key=lambda ind: ind.fitness.values[0])
            eval_team.append(best_ind)
        return eval_team

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team
        eval_team = self.formEvaluationTeam(population)
        # Evaluate that team however many times we are evaluating teams
        eval_teams = [eval_team for _ in range(self.num_evaluations_per_team)]
        return self.evaluateTeams(eval_teams)

    def formTeams(self, population, inds=None):
        # Start a list of teams
        teams = []

        if inds is None:
            team_inds = range(self.subpopulation_size)
        else:
            team_inds = inds

        # For each individual in a subpopulation
        for i in team_inds:
            # Make a team
            team = []

            # For each subpopulation in the population
            for subpop in population:
                # Put the i'th indiviudal on the team
                team.append(subpop[i])

            # Need to save that team for however many evaluations
            # we're doing per team
            for _ in range(self.num_evaluations_per_team):
                # Save that team
                teams.append(team)

        return teams

    def evaluateTeams(self, teams):
        if self.use_multiprocessing:
            jobs = self.map(self.evaluateTeam, teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.map(self.evaluateTeam, teams))
        return eval_infos

    def evaluateTeam(self, team, compute_team_fitness=True):
        # Set up models
        rover_nns = [deepcopy(self.rover_nn_template) for _ in range(self.num_rovers)]

        # Load in the weights
        for rover_nn, individual in zip(rover_nns, team[: self.num_rovers]):
            rover_nn.set_params(torch.tensor(individual, dtype=torch.double).to(DEVICE))

        agent_nns = rover_nns

        # Set up the enviornment
        env = createEnv(self.config)

        observations, _ = env.reset()

        agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
        poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

        observations_arrs = []

        for observation in observations:
            observation_arr = []

            for i in range(len(observation)):
                observation_arr.append(observation(i))

            observation_arr = np.array(observation_arr, dtype=np.float64)
            observations_arrs.append(observation_arr)

        joint_state_trajectory = [agent_positions + poi_positions]
        joint_observation_trajectory = [observations_arrs]
        joint_action_trajectory = []
        rewards_list = []
        G_list = []

        for _ in range(self.num_steps):

            # Compute the actions of all rovers
            observation_arrs = []
            actions_arrs = []
            actions = []

            for ind, (observation, agent_nn) in enumerate(zip(observations, agent_nns)):

                obs_tensor = observation.data()
                obs_tensor.reshape((8,))  # State space is 8 dimensional
                obs_tensor = np.frombuffer(obs_tensor, dtype=np.double, count=8)
                obs_tensor = torch.from_numpy(obs_tensor).to(DEVICE)

                action = agent_nn.forward(obs_tensor.unsqueeze(0))

                action_arr = action.squeeze().cpu().detach().numpy()

                # Multiply by agent velocity
                if ind <= self.num_rovers:
                    action_arr *= self.config["ccea"]["model"]["rover_max_velocity"]

                # Save this info for debugging purposes
                observation_arrs.append(observation_arr)
                actions_arrs.append(action_arr)

            for action_arr in actions_arrs:
                action = rovers.tensor(action_arr)
                actions.append(action)

            observations, rewards = env.step(actions)

            # Get all the states and all the actions of all agents
            agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
            poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

            joint_observation_trajectory.append(observation_arrs)
            joint_action_trajectory.append(actions_arrs)
            joint_state_trajectory.append(agent_positions + poi_positions)

            # Store rewards of every episode
            rewards_list.append(list(rewards))

            # Append G for the episode to list
            agent_pack = rovers.AgentPack(agent_index=0, agents=env.rovers(), entities=env.pois())

            G_list.append(rovers.rewards.Global().compute(agent_pack))

        if compute_team_fitness:

            match (self.fitness_method):

                case "aggregate":
                    rewards = np.sum(np.array(rewards_list), axis=0)
                    team_fitness = sum(G_list)

                case "last_step":
                    team_fitness = G_list[-1]

            fitnesses = tuple([(r,) for r in rewards] + [(team_fitness,)])

        else:
            # Each index corresponds to an agent's rewards
            # We only evaulate the team fitness based on the last step
            # so we only keep the last set of rewards generated by the team
            fitnesses = tuple([(r,) for r in rewards])

        return EvalInfo(
            fitnesses=fitnesses,
            joint_trajectory=JointTrajectory(
                joint_state_trajectory,
                joint_observation_trajectory,
                joint_action_trajectory,
            ),
        )

    def mutateIndividual(self, individual):
        return tools.mutGaussian(
            individual,
            mu=self.config["ccea"]["mutation"]["mean"],
            sigma=self.config["ccea"]["mutation"]["std_deviation"],
            indpb=self.config["ccea"]["mutation"]["independent_probability"],
        )

    def mutate(self, population):
        # Don't mutate the elites from n-elites
        for num_individual in range(self.num_mutants):
            mutant_id = num_individual + self.n_elites
            for subpop in population:
                self.mutateIndividual(subpop[mutant_id])
                del subpop[mutant_id].fitness.values

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        offspring = tools.selBest(subpopulation, self.n_elites)
        if self.include_elites_in_tournament:
            offspring += tools.selTournament(subpopulation, len(subpopulation) - self.n_elites, tournsize=2)
        else:
            # Get the remaining worse individuals
            remaining_offspring = tools.selWorst(subpopulation, len(subpopulation) - self.n_elites)
            # Add those remaining individuals through a binary tournament
            offspring += tools.selTournament(remaining_offspring, len(remaining_offspring), tournsize=2)

        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [deepcopy(individual) for individual in offspring]

    def select(self, population):
        # Offspring is a list of subpopulation
        offspring = []
        # For each subpopulation in the population
        for subpop in population:
            # Perform a selection on that subpopulation and add it to the offspring population
            offspring.append(self.selectSubPopulation(subpop))
        return offspring

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def assignFitnesses(self, teams, eval_infos):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        if self.num_evaluations_per_team == 1:
            for team, eval_info in zip(teams, eval_infos):
                fitnesses = eval_info.fitnesses
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit
        else:
            team_list = []
            eval_info_list = []
            for team, eval_info in zip(teams, eval_infos):
                team_list.append(team)
                eval_info_list.append(eval_info)

            for team_id, team in enumerate(team_list[:: self.num_evaluations_per_team]):
                # Get all the eval infos for this team
                team_eval_infos = eval_info_list[
                    team_id * self.num_evaluations_per_team : (team_id + 1) * self.num_evaluations_per_team
                ]
                # Aggregate the fitnesses into a big numpy array
                all_fitnesses = [eval_info.fitnesses for eval_info in team_eval_infos]
                average_fitnesses = [0 for _ in range(len(all_fitnesses[0]))]
                for fitnesses in all_fitnesses:
                    for count, fit in enumerate(fitnesses):
                        average_fitnesses[count] += fit[0]
                for ind in range(len(average_fitnesses)):
                    average_fitnesses[ind] = average_fitnesses[ind] / self.num_evaluations_per_team
                # And now get that back to the individuals
                fitnesses = tuple([(f,) for f in average_fitnesses])
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, trial_dir):
        eval_fitness_dir = trial_dir / "fitness.csv"
        header = "generation,team_fitness_aggregated"
        for j in range(self.num_rovers):
            header += ",rover_" + str(j)

        for i in range(self.num_evaluations_per_team):
            header += ",team_fitness_" + str(i)
            for j in range(self.num_rovers):
                header += ",team_" + str(i) + "_rover_" + str(j)

        header += "\n"
        with open(eval_fitness_dir, "w") as file:
            file.write(header)

    def writeEvalFitnessCSV(self, trial_dir, eval_infos):
        eval_fitness_dir = trial_dir / "fitness.csv"
        gen = str(self.gen)

        if len(eval_infos) == 1:
            eval_info = eval_infos[0]
            team_fit = str(eval_info.fitnesses[-1][0])
            agent_fits = [str(fit[0]) for fit in eval_info.fitnesses[:-1]]
            fit_list = [gen, team_fit] + agent_fits
            fit_str = ",".join(fit_list) + "\n"

        else:
            team_eval_infos = []
            for eval_info in eval_infos:
                team_eval_infos.append(eval_info)

            # Aggergate the fitnesses into a big numpy array
            num_ind_per_team = len(team_eval_infos[0].fitnesses)
            all_fit = np.zeros(shape=(self.num_evaluations_per_team, num_ind_per_team))

            for num_eval, eval_info in enumerate(team_eval_infos):
                fitnesses = eval_info.fitnesses
                for num_ind, fit in enumerate(fitnesses):
                    all_fit[num_eval, num_ind] = fit[0]
                all_fit[num_eval, -1] = fitnesses[-1][0]

            # Now compute a sum/average/min/etc dependending on what config specifies
            agg_fit = np.average(all_fit, axis=0)
            # And now record it all, starting with the aggregated one
            agg_team_fit = str(agg_fit[-1])
            agg_agent_fits = [str(fit) for fit in agg_fit[:-1]]
            fit_str = gen + "," + ",".join([agg_team_fit] + agg_agent_fits) + ","
            # And now add all the fitnesses from individual trials
            # Each row should have the fitnesses for an evaluation
            for row in all_fit:
                team_fit = str(row[-1])
                agent_fits = [str(fit) for fit in row[:-1]]
                fit_str += ",".join([team_fit] + agent_fits)
            fit_str += "\n"
        # Now save it all to the csv
        with open(eval_fitness_dir, "a") as file:
            file.write(fit_str)

    def writeEvalTrajs(self, trial_dir, eval_infos):
        gen_folder_name = "gen_" + str(self.gen)
        gen_dir = trial_dir / gen_folder_name

        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

        for eval_id, eval_info in enumerate(eval_infos):
            eval_filename = "eval_team_" + str(eval_id) + "_joint_traj.csv"
            eval_dir = gen_dir / eval_filename
            with open(eval_dir, "w") as file:
                # Build up the header (labels at the top of the csv)
                header = ""
                # First the states (agents and POIs)
                for i in range(self.num_rovers):
                    header += "rover_" + str(i) + "_x,rover_" + str(i) + "_y,"

                for i in range(self.num_pois):
                    header += "rover_poi_" + str(i) + "_x,rover_poi_" + str(i) + "_y,"

                # Observations
                for i in range(self.num_rovers):
                    for j in range(self.num_rover_sectors * 2):
                        header += "rover_" + str(i) + "_obs_" + str(j) + ","

                # Actions
                for i in range(self.num_rovers):
                    header += "rover_" + str(i) + "_dx,rover_" + str(i) + "_dy,"

                header += "\n"
                # Write out the header at the top of the csv
                file.write(header)

                # Now fill in the csv with the data
                # One line at a time
                joint_traj = eval_info.joint_trajectory

                # We're going to pad the actions with None because
                # the agents cannot take actions at the last timestep, but
                # there is a final joint state/observations
                action_padding = []
                for action in joint_traj.actions[0]:
                    action_padding.append([None for _ in action])
                joint_traj.actions.append(action_padding)
                for joint_state, joint_observation, joint_action in zip(
                    joint_traj.states, joint_traj.observations, joint_traj.actions
                ):
                    # Aggregate state info
                    state_list = []
                    for state in joint_state:
                        state_list += [str(state_var) for state_var in state]
                    state_str = ",".join(state_list)
                    # Aggregate observation info
                    observation_list = []
                    for observation in joint_observation:
                        observation_list += [str(obs_val) for obs_val in observation]
                    observation_str = ",".join(observation_list)
                    # Aggregate action info
                    action_list = []
                    for action in joint_action:
                        action_list += [str(act_val) for act_val in action]
                    action_str = ",".join(action_list)
                    # Put it all together
                    csv_line = state_str + "," + observation_str + "," + action_str + "\n"
                    # Write it out
                    file.write(csv_line)

    def run(self):
        for num_trial in range(self.config["experiment"]["num_trials"]):
            # Init gen counter
            self.gen = 0

            # Create directory for saving data
            trial_dir = self.trials_dir / ("trial_" + str(num_trial))
            if not os.path.isdir(trial_dir):
                os.makedirs(trial_dir)

            # Create csv file for saving evaluation fitnesses
            self.createEvalFitnessCSV(trial_dir)

            # Initialize the population
            pop = self.population()

            # Create the teams
            teams = self.formTeams(pop)

            # Evaluate the teams
            eval_infos = self.evaluateTeams(teams)

            # Assign fitnesses to individuals
            self.assignFitnesses(teams, eval_infos)

            # Evaluate a team with the best indivdiual from each subpopulation
            eval_infos = self.evaluateEvaluationTeam(pop)

            # Save fitnesses of the evaluation team
            self.writeEvalFitnessCSV(trial_dir, eval_infos)

            # Save trajectories of evaluation team
            if self.save_trajectories:
                self.writeEvalTrajs(trial_dir, eval_infos)

            for gen in tqdm(range(self.config["ccea"]["num_generations"])):
                # Update gen counter
                self.gen = gen + 1

                # Perform selection
                offspring = self.select(pop)

                # Perform mutation
                self.mutate(offspring)

                # Shuffle subpopulations in offspring
                # to make teams random
                self.shuffle(offspring)

                # Form teams for evaluation
                teams = self.formTeams(offspring)

                # Evaluate each team
                eval_infos = self.evaluateTeams(teams)

                # Now assign fitnesses to each individual
                self.assignFitnesses(teams, eval_infos)

                # Evaluate a team with the best indivdiual from each subpopulation
                eval_infos = self.evaluateEvaluationTeam(offspring)

                # Save fitnesses
                self.writeEvalFitnessCSV(trial_dir, eval_infos)

                # Save trajectories
                if self.save_trajectories and self.gen % self.num_gens_between_save_traj == 0:
                    self.writeEvalTrajs(trial_dir, eval_infos)

                # Now populate the population with individuals from the offspring
                self.setPopulation(pop, offspring)

        if self.use_multiprocessing:
            self.pool.close()

        return pop


def runCCEA(config_dir):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir)
    return ccea.run()
