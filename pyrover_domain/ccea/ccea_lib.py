from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm
import torch

import torch.multiprocessing as mp
import random

from pyrover_domain.models.mlp import MLP_Policy
from pyrover_domain.models.gru import GRU_Policy
from pyrover_domain.models.cnn import CNN_Policy

from pyrover_domain.librovers import rovers
from pyrover_domain.custom_env import createEnv
from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging
import pickle

import uuid

from itertools import combinations

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Agent(object):
    def __init__(self, idx: int, parameters: list):
        self.idx = idx
        self.parameters = parameters


class Team(object):
    def __init__(self, idx: int, individuals: list[Agent] = None, combination: list = None):
        self.idx = idx
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []


class JointTrajectory(object):
    def __init__(
        self,
        joint_state_trajectory,
        joint_observation_trajectory,
        joint_action_trajectory,
    ):
        self.states = joint_state_trajectory
        self.observations = joint_observation_trajectory
        self.actions = joint_action_trajectory


class EvalInfo(object):
    def __init__(self, team_id, team_formation, agent_fitnesses, team_fitness, joint_trajectory):
        self.team_id = team_id
        self.team_formation = team_formation
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness
        self.joint_trajectory = joint_trajectory


class CooperativeCoevolutionaryAlgorithm:
    def __init__(self, config_dir):
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = f"{Path(self.config_dir).parents[1]}/results"

        with open(str(self.config_dir), "r") as file:
            self.config = yaml.safe_load(file)

        # Experiemtn data
        self.trial_name = self.config["experiment"]["trial_name"]

        # Start by setting up variables for different agents
        self.num_rovers = len(self.config["env"]["rovers"])
        self.use_teaming = self.config["teaming"]["use_teaming"]

        self.n_eval_per_team = self.config["ccea"]["evaluation"]["multi_evaluation"]["num_evaluations"]
        self.team_size = self.config["teaming"]["team_size"] if self.use_teaming else self.num_rovers
        self.team_combinations = [combo for combo in combinations(range(self.num_rovers), self.team_size)]

        self.n_eval_per_team_set = (
            len(self.team_combinations) * self.n_eval_per_team if self.use_teaming else self.n_eval_per_team
        )

        self.subpopulation_size = self.config["ccea"]["population"]["subpopulation_size"]

        self.num_hidden = self.config["ccea"]["model"]["hidden_layers"]
        self.model_type = self.config["ccea"]["model"]["type"]
        self.weight_initialization = self.config["ccea"]["weight_initialization"]

        self.sensor_type = self.config["env"]["rovers"][0]["sensor_type"]
        self.image_size = self.config["env"]["img_sensor_size"]

        self.n_pois = len(self.config["env"]["pois"])

        self.n_elites = round(self.config["ccea"]["selection"]["n_elites"] * self.subpopulation_size)
        self.n_mutants = self.subpopulation_size - self.n_elites
        self.n_extinction_candidates = round(
            self.config["ccea"]["selection"]["n_extinction_candidates"] * self.n_mutants
        )

        self.aggregation_method = self.config["ccea"]["evaluation"]["multi_evaluation"]["aggregation_method"]
        self.fitness_method = self.config["ccea"]["evaluation"]["fitness_method"]

        self.n_steps = self.config["ccea"]["num_steps"]
        self.n_gens = self.config["ccea"]["num_generations"]

        self.n_rover_sectors = int(360 / self.config["env"]["rovers"][0]["resolution"])
        self.rover_nn_template = self.generateTemplateNN()
        self.rover_nn_size = self.rover_nn_template.num_params

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data loading
        self.load_checkpoint = self.config["data"]["load_checkpoint"]
        self.load_checkpoint_filename = self.config["data"]["load_checkpoint_filename"]

        # Data saving variables
        self.save_trajectories = self.config["data"]["save_trajectories"]
        self.save_checkpoint = self.config["data"]["save_checkpoint"]
        self.num_gens_between_save = self.config["data"]["num_gens_between_save"]

        # Create the type of fitness we're optimizing
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "subpopulation",
            tools.initRepeat,
            list,
            self.createIndividual,
            n=self.config["ccea"]["population"]["subpopulation_size"],
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.subpopulation,
            n=self.num_rovers,
        )

        # Set up multi processing methods
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

    def createIndividual(self):
        match (self.weight_initialization):
            case "kaiming":
                temp_model = self.generateTemplateNN()
                params = temp_model.get_params()
        return creator.Individual(params[:].cpu().numpy().astype(np.float64))

    def generateTemplateNN(self):
        match (self.model_type):

            case "GRU":
                agent_nn = GRU_Policy(
                    input_size=2 * self.n_rover_sectors,
                    hidden_size=self.num_hidden[0],
                    output_size=2,
                    num_layers=1,
                ).to(DEVICE)

            case "CNN":
                agent_nn = CNN_Policy(
                    img_size=self.image_size,
                ).to(DEVICE)

            case "MLP":
                agent_nn = MLP_Policy(
                    input_size=2 * self.n_rover_sectors,
                    hidden_layers=len(self.num_hidden),
                    hidden_size=self.num_hidden[0],
                    output_size=2,
                ).to(DEVICE)

        return agent_nn

    def getBestAgents(self, population) -> list[Agent]:
        best_agents = []

        # Get best agents
        for idx, subpop in enumerate(population):
            # Get the best N individuals
            best_ind = tools.selBest(subpop, 1)[0]
            best_agents.append(Agent(idx=idx, parameters=best_ind))

        return best_agents

    def evaluateBestTeam(self, population):
        # Create evaluation teams
        eval_teams = self.formTeams(population, for_evaluation=True)
        return self.evaluateTeams(eval_teams)

    def formTeams(self, population, for_evaluation: bool = False) -> list[Team]:
        # Start a list of teams
        teams = []

        if for_evaluation:
            joint_policies = 1
        else:
            joint_policies = self.subpopulation_size

        # For each individual in a subpopulation
        for i in range(joint_policies):

            if for_evaluation:
                agents = self.getBestAgents(population)
            else:
                # Get agents in this row of subpopulations
                agents = [Agent(idx=idx, parameters=subpop[i]) for idx, subpop in enumerate(population)]

            if self.use_teaming:
                # Put the i'th individual on the team if it is inside our team combinations
                for combination in self.team_combinations:

                    teams.extend(
                        [
                            Team(idx=i, individuals=[agents[idx] for idx in combination], combination=combination)
                            for _ in range(self.n_eval_per_team)
                        ]
                    )

            else:
                team = Team(idx=i)

                team.individuals = agents
                team.combination = [agent.idx for agent in agents]

                # Need to save that team for however many evaluations
                # we're doing per team
                teams.extend([team for _ in range(self.n_eval_per_team)])

        return teams

    def evaluateTeams(self, teams: list[Team]):
        if self.use_multiprocessing:
            jobs = self.map(self.evaluateTeam, teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.map(self.evaluateTeam, teams))
        return eval_infos

    def evaluateTeam(self, team: Team):
        # Set up models
        rover_nns = [deepcopy(self.rover_nn_template) for _ in range(self.team_size)]

        # Load in the weights
        for rover_nn, individual in zip(rover_nns, team.individuals):
            rover_nn.set_params(torch.from_numpy(individual.parameters).to(DEVICE))

        agent_nns = rover_nns

        # Set up the enviornment
        env = createEnv(self.config)

        # Get initial observations per agent
        observations, _ = env.reset()

        agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
        poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

        # Change observations type to numpy
        observations_arrs = []

        for observation in observations:
            observation_arr = []

            for i in range(len(observation)):
                observation_arr.append(observation(i))

            observation_arr = np.array(observation_arr, dtype=np.float64)
            observations_arrs.append(observation_arr)

        # Set evaluation storage variables
        joint_state_trajectory = [agent_positions + poi_positions]
        joint_observation_trajectory = [observations_arrs]
        joint_action_trajectory = []
        rewards_list = []
        G_list = []

        # Start evaluation
        for _ in range(self.n_steps):

            # Compute the actions of all rovers
            observation_arrs = []
            actions_arrs = []

            for _, (observation, agent_nn) in enumerate(zip(observations, agent_nns)):

                obs_tensor = observation.data()

                # State pre processing based on sensor type
                match (self.sensor_type):

                    case "lidar":
                        obs_tensor.reshape((self.n_rover_sectors * 2,))  # State space is 8 dimensional
                        obs_tensor = np.frombuffer(obs_tensor, dtype=np.float64, count=8)

                    case "camera":
                        flat_img_size = int(np.pow(self.image_size, 2) * 2)
                        obs_tensor.reshape((flat_img_size,))
                        obs_tensor = np.frombuffer(obs_tensor, dtype=np.float64, count=flat_img_size)
                        obs_tensor = np.reshape(
                            obs_tensor, (2, self.image_size, self.image_size)
                        )  # 2 channels since we have a POI image and a Rover Image

                obs_tensor = torch.from_numpy(obs_tensor).to(DEVICE)

                action = agent_nn.forward(obs_tensor.unsqueeze(0))

                action_arr = action.squeeze().cpu().detach().numpy()

                # Multiply by agent velocity
                action_arr *= self.config["ccea"]["model"]["rover_max_velocity"]

                # Save this info for debugging purposes
                observation_arrs.append(observation_arr)
                actions_arrs.append(action_arr)

            actions = [rovers.tensor(action_arr) for action_arr in actions_arrs]

            observations, rewards = env.step(actions)

            # Get all the states and all the actions of all agents
            agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
            poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

            # Store joint actions and states
            joint_observation_trajectory.append(observation_arrs)
            joint_action_trajectory.append(actions_arrs)
            joint_state_trajectory.append(agent_positions + poi_positions)

            # Store rewards of every episode
            rewards_list.append(list(rewards))

            # Append G for the episode to list
            agent_pack = rovers.AgentPack(agent_index=0, agents=env.rovers(), entities=env.pois())

            G_list.append(rovers.rewards.Global().compute(agent_pack))

        # Compute team fitness
        match (self.fitness_method):

            case "aggregate":
                rewards = np.sum(np.array(rewards_list), axis=0)
                team_fitness = sum(G_list)

            case "last_step":
                rewards = np.array(list(rewards))
                team_fitness = G_list[-1]

        return EvalInfo(
            team_id=team.idx,
            team_formation=team.combination,
            agent_fitnesses=tuple(zip(team.combination, rewards)),
            team_fitness=team_fitness,
            joint_trajectory=JointTrajectory(
                joint_state_trajectory,
                joint_observation_trajectory,
                joint_action_trajectory,
            ),
        )

    def mutateIndividual(self, individual):

        min_stdev = self.config["ccea"]["mutation"]["min_std_deviation"]
        max_stdev = self.config["ccea"]["mutation"]["max_std_deviation"]
        mut_lifespan = self.config["ccea"]["mutation"]["lifespan"]
        use_decay = self.config["ccea"]["mutation"]["use_decay"]
        mu = self.config["ccea"]["mutation"]["mean"]
        indpb = self.config["ccea"]["mutation"]["independent_probability"]

        if use_decay:
            mutation_lifespan = int(self.n_gens * mut_lifespan)

            decay_rate = np.log(min_stdev / max_stdev) / mutation_lifespan

            mutation_stdev = max_stdev * np.exp(decay_rate * self.gen)

            if mutation_stdev < min_stdev:
                mutation_stdev = min_stdev

        else:
            mutation_stdev = min_stdev

        return tools.mutGaussian(
            individual,
            mu=mu,
            sigma=mutation_stdev,
            indpb=indpb,
        )

    def mutate(self, population):
        # Don't mutate the elites
        for num_individual in range(self.n_mutants):

            mutant_idx = num_individual + self.n_elites

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness.values = (np.float64(0.0),)

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        elites = tools.selBest(subpopulation, self.n_elites)

        # Get the remaining worse individuals
        # non_elites = tools.selWorst(subpopulation, len(subpopulation) - self.n_elites)

        # if self.config["ccea"]["selection"]["use_extinction"]:
        #     # Select extinction candidates
        #     extinction_candidates = tools.selWorst(non_elites, self.n_extinction_candidates)

        #     # Reinitialize extinction candidates
        #     ext_prob = self.config["ccea"]["selection"]["extinction_prob"]
        #     for idx, _ in enumerate(extinction_candidates):
        #         if np.random.choice(
        #             [True, False],
        #             1,
        #             p=[
        #                 ext_prob,
        #                 1 - ext_prob,
        #             ],
        #         ):
        #             non_elites[idx] = self.createIndividual()

        non_elites = tools.selTournament(subpopulation, len(subpopulation) - self.n_elites, tournsize=2)

        offspring = elites + non_elites

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

    def assignFitnesses(self, teams: list[Team], eval_infos: list[EvalInfo]):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        if self.n_eval_per_team_set == 1:
            for team, eval_info in zip(teams, eval_infos):
                for individual, fit in zip(team.individuals, eval_info.agent_fitnesses[1]):
                    individual.parameters.fitness.values = fit
        else:

            for team_id, team in enumerate(teams[:: self.n_eval_per_team_set]):

                # Get all the eval infos for this team
                team_eval_infos = eval_infos[
                    team_id * self.n_eval_per_team_set : (team_id + 1) * self.n_eval_per_team_set
                ]

                # Aggregate the fitnesses into a big numpy array
                accumulated_fitnesses = [0 for _ in range(self.num_rovers)]

                for eval_info in team_eval_infos:
                    for idx, fit in eval_info.agent_fitnesses:
                        accumulated_fitnesses[idx] += fit

                # Put into tuples due to deap.base.Fitness type
                for idx, individual in zip(team.combination, team.individuals):
                    individual.parameters.fitness.values = (accumulated_fitnesses[idx],)

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, trial_dir):
        eval_fitness_dir = f"{trial_dir}/fitness.csv"
        header = "generation,team_fitness_aggregated"
        for j in range(self.num_rovers):
            header += ",rover_" + str(j)

        for i in range(self.n_eval_per_team_set):
            header += ",team_fitness_" + str(i)
            for j in range(self.num_rovers):
                header += ",team_" + str(i) + "_rover_" + str(j)

        header += "\n"
        with open(eval_fitness_dir, "w") as file:
            file.write(header)

    def writeEvalFitnessCSV(self, trial_dir, eval_infos):
        eval_fitness_dir = f"{trial_dir}/fitness.csv"
        gen = str(self.gen)

        if len(eval_infos) == 1:
            eval_info = eval_infos[0]
            team_fit = str(eval_info.team_fitness)
            agent_fits = [str(fit[1]) for fit in eval_info.agent_fitnesses]
            fit_list = [gen, team_fit] + agent_fits
            fit_str = ",".join(fit_list) + "\n"

        else:

            # Aggergate the fitnesses into a big numpy array
            num_ind_per_team = len(eval_infos[0].agent_fitnesses) + 1  # +1 to include team fitness
            all_fit = np.zeros(shape=(self.n_eval_per_team_set, num_ind_per_team))

            for num_eval, eval_info in enumerate(eval_infos):
                for num_ind, fit in enumerate(eval_info.agent_fitnesses):
                    all_fit[num_eval, num_ind] = fit[1]
                all_fit[num_eval, -1] = eval_info.team_fitness

            # Now compute a sum/average/min/etc dependending on what config specifies
            agg_fit = np.sum(all_fit, axis=0)

            # And now record it all, starting with the aggregated one
            agg_team_fit = str(agg_fit[-1])
            agg_agent_fits = [str(fit) for fit in agg_fit[:-1]]
            fit_str = f"{gen},{','.join([agg_team_fit] + agg_agent_fits)},"

            # And now add all the fitnesses from individual trials
            # Each row should have the fitnesses for an evaluation
            for row in all_fit:
                team_fit = str(row[-1])
                agent_fits = [str(fit) for fit in row[:-1]]
                fit_str += f"{','.join([team_fit] + agent_fits)},"
            fit_str += "\n"

        # Now save it all to the csv
        with open(eval_fitness_dir, "a") as file:
            file.write(fit_str)

    def writeEvalTrajs(self, trial_dir, eval_infos):
        gen_folder_name = f"gen_{self.gen}"
        gen_dir = f"{trial_dir}/{gen_folder_name}"

        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

        for eval_id, eval_info in enumerate(eval_infos):

            eval_filename = f"eval_team_{eval_id}_joint_traj.csv"
            eval_dir = f"{gen_dir}/{eval_filename}"

            with open(eval_dir, "w") as file:
                # Build up the header (labels at the top of the csv)
                header = ""

                # The team formation
                header += f"team_formation,"

                # The states (agents and POIs)
                for i in range(self.team_size):
                    header += f"rover_{i}_x,rover_{i}_y,"

                for i in range(self.n_pois):
                    header += f"rover_poi_{i}_x,rover_poi_{i}_y,"

                # Observations
                for i in range(self.team_size):
                    for j in range(self.n_rover_sectors * 2):
                        header += f"rover_{i}_obs_{i},"

                # Actions
                for i in range(self.team_size):
                    header += f"rover_{i}_dx,rover_{i}_dy,"

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
                    team_formation_str = "-".join([str(i) for i in eval_info.team_formation])
                    csv_line = f"{team_formation_str},{state_str},{observation_str},{action_str}\n"
                    # Write it out
                    file.write(csv_line)

    def run(self):
        # Load checkpoint
        if self.load_checkpoint:
            with open(trial_dir / self.load_checkpoint_filename, "rb") as handle:
                checkpoint = pickle.load(handle)
                pop = checkpoint["population"]
                self.checkpoint_gen = checkpoint["gen"]
                self.checkpoint_trial = checkpoint["trial"]
            logging.info(f"Loading GEN:{self.checkpoint_gen} TRIAL: {self.checkpoint_trial}...")
        else:
            logging.info(f"Starting run...")

        for n_trial in range(self.config["experiment"]["num_trials"]):

            # Create directory for saving data
            trial_dir = f"{self.trials_dir}/trial_{str(n_trial)}_{self.trial_name}"
            if not os.path.isdir(trial_dir):
                os.makedirs(trial_dir)

            # Initialize the population or load models
            if self.load_checkpoint and n_trial <= self.checkpoint_trial:
                continue
            else:
                pop = self.toolbox.population()

                # Create csv file for saving evaluation fitnesses
                self.createEvalFitnessCSV(trial_dir)

            for n_gen in tqdm(range(self.n_gens + 1)):

                # Set gen counter global var
                self.gen = n_gen

                # Get loading bar up to checkpoint
                if self.load_checkpoint and n_gen <= self.checkpoint_gen:
                    continue

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

                # Evaluate a team with the best individual from each subpopulation
                eval_infos = self.evaluateBestTeam(offspring)

                # Now populate the population with individuals from the offspring
                self.setPopulation(pop, offspring)

                # Save fitnesses
                self.writeEvalFitnessCSV(trial_dir, eval_infos)

                # Save trajectories and checkpoint
                if n_gen % self.num_gens_between_save == 0:

                    if self.save_trajectories:
                        # Save trajectories
                        self.writeEvalTrajs(trial_dir, eval_infos)

                    if self.save_checkpoint:
                        trial_done = 1 if n_gen == self.n_gens else 0
                        with open(f"{trial_dir}/{self.load_checkpoint_filename}", "wb") as handle:
                            pickle.dump(
                                {"population": pop, "gen": n_gen, "trial": n_trial + trial_done},
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
        if self.use_multiprocessing:
            self.pool.close()

        return pop


def runCCEA(config_dir):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir)
    return ccea.run()
