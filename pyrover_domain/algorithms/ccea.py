from deap import base
from deap import creator
from deap import tools
import torch

import torch.multiprocessing as mp
import random

from pyrover_domain.policies.mlp import MLP_Policy
from pyrover_domain.policies.gru import GRU_Policy
from pyrover_domain.policies.cnn import CNN_Policy

from pyrover_domain.fitness_critic.fitness_critic import FitnessCritic

from pyrover_domain.librovers import rovers
from pyrover_domain.custom_env import createEnv
from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging
import pickle
import csv

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


class JointState(object):
    def __init__(
        self,
    ):
        self.agent_positions = []
        self.poi_positions = []


class JointTrajectory(object):
    def __init__(
        self,
        joint_state_traj: JointState,
        joint_obs_traj,
        joint_act_traj,
    ):
        self.states = joint_state_traj
        self.observations = joint_obs_traj
        self.actions = joint_act_traj


class EvalInfo(object):
    def __init__(self, team_id, team_formation, agent_fitnesses, team_fitness, joint_traj):
        self.team_id = team_id
        self.team_formation = team_formation
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness
        self.joint_traj = joint_traj


class CooperativeCoevolutionaryAlgorithm:
    def __init__(self, config_dir, experiment_name: str, trial_id: int, load_checkpoint: bool):

        self.trial_id = trial_id
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = os.path.join(self.config_dir.parents[2], "results", experiment_name)

        with open(str(self.config_dir), "r") as file:
            self.config = yaml.safe_load(file)

        # Experiment data
        self.trial_name = Path(self.config_dir).stem

        # Start by setting up variables for different agents
        self.num_rovers = len(self.config["env"]["rovers"])
        self.use_teaming = self.config["teaming"]["use_teaming"]
        self.use_fc = self.config["fitness_critic"]["use_fit_crit"]
        self.fit_crit_loss_type = self.config["fitness_critic"]["loss_type"]

        self.n_eval_per_team = self.config["ccea"]["evaluation"]["num_evaluations"]
        self.team_size = self.config["teaming"]["team_size"] if self.use_teaming else self.num_rovers
        self.team_combinations = [combo for combo in combinations(range(self.num_rovers), self.team_size)]

        self.n_eval_per_team_set = (
            len(self.team_combinations) * self.n_eval_per_team if self.use_teaming else self.n_eval_per_team
        )

        self.subpopulation_size = self.config["ccea"]["population"]["subpopulation_size"]

        self.policy_num_hidden = self.config["ccea"]["policy"]["hidden_layers"]
        self.policy_type = self.config["ccea"]["policy"]["type"]
        self.weight_initialization = self.config["ccea"]["weight_initialization"]

        self.fit_crit_type = self.config["fitness_critic"]["type"]
        self.fit_crit_num_hidden = self.config["fitness_critic"]["hidden_layers"]

        self.sensor_type = self.config["env"]["rovers"][0]["sensor_type"]
        self.image_size = self.config["env"]["img_sensor_size"]

        self.n_pois = len(self.config["env"]["pois"])

        self.n_elites = round(self.config["ccea"]["selection"]["n_elites"] * self.subpopulation_size)
        self.n_mutants = self.subpopulation_size - self.n_elites

        self.fitness_method = self.config["ccea"]["evaluation"]["fitness_method"]

        self.n_steps = self.config["ccea"]["num_steps"]
        self.n_gens = self.config["ccea"]["num_generations"]

        self.n_rover_sectors = int(360 / self.config["env"]["rovers"][0]["resolution"])
        self.rover_nn_template = self.generateTemplateNN()
        self.rover_nn_size = self.rover_nn_template.num_params

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data loading
        self.load_checkpoint = load_checkpoint

        # Data saving variables
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
        match (self.policy_type):

            case "GRU":
                agent_nn = GRU_Policy(
                    input_size=2 * self.n_rover_sectors,
                    hidden_size=self.policy_num_hidden[0],
                    output_size=2,
                    n_layers=1,
                ).to(DEVICE)

            case "CNN":
                agent_nn = CNN_Policy(
                    img_size=self.image_size,
                ).to(DEVICE)

            case "MLP":
                agent_nn = MLP_Policy(
                    input_size=2 * self.n_rover_sectors,
                    hidden_layers=len(self.policy_num_hidden),
                    hidden_size=self.policy_num_hidden[0],
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

        # For each row in the population of subpops (grabs an individual from each row in the subpops)
        for i in range(joint_policies):

            if for_evaluation:
                agents = self.getBestAgents(population)
            else:
                # Get agents in this row of subpopulations
                agents = [Agent(idx=idx, parameters=subpop[i]) for idx, subpop in enumerate(population)]

            # Put the i'th individual on the team if it is inside our team combinations
            for combination in self.team_combinations:

                teams.extend(
                    [
                        Team(idx=i, individuals=[agents[idx] for idx in combination], combination=combination)
                        for _ in range(self.n_eval_per_team)
                    ]
                )

        return teams

    def process_observation(self, observation):
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

        return obs_tensor

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

        # Process initial observations
        observations_arrs = [self.process_observation(obs) for obs in observations]

        # Set evaluation storage variables
        joint_state_traj = JointState()
        joint_state_traj.agent_positions.append(agent_positions)
        joint_state_traj.poi_positions.append(poi_positions)
        joint_obs_traj = [observations_arrs]
        joint_act_traj = []
        rewards_list = []
        G_list = []

        # Start evaluation
        for _ in range(self.n_steps):

            # Compute the actions of all rovers
            observation_arrs = []
            actions_arrs = []

            for observation, agent_nn in zip(observations, agent_nns):

                obs_array = self.process_observation(observation)

                obs_tensor = torch.from_numpy(obs_array).to(DEVICE)

                action = agent_nn.forward(obs_tensor.unsqueeze(0))

                action_arr = action.squeeze().cpu().detach().numpy()

                # Multiply by agent velocity
                action_arr *= self.config["ccea"]["policy"]["rover_max_velocity"]

                # Save this info for debugging purposes
                observation_arrs.append(obs_array)
                actions_arrs.append(action_arr)

            actions = [rovers.tensor(action_arr) for action_arr in actions_arrs]

            observations, rewards = env.step(actions)

            # Get all the states and all the actions of all agents
            agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
            poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

            # Store joint actions and states
            joint_obs_traj.append(observation_arrs)
            joint_act_traj.append(actions_arrs)
            joint_state_traj.agent_positions.append(agent_positions)
            joint_state_traj.poi_positions.append(poi_positions)

            # Store rewards of every episode (could be G or D depending on yaml config)
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
            joint_traj=JointTrajectory(
                joint_state_traj,
                np.array(joint_obs_traj),
                joint_act_traj,
            ),
        )

    def mutateIndividual(self, individual):

        individual *= np.random.normal(
            loc=self.config["ccea"]["mutation"]["mean"],
            scale=self.config["ccea"]["mutation"]["std_deviation"],
            size=np.shape(individual),
        )

    def mutate(self, population):
        # Don't mutate the elites
        for num_individual in range(self.n_mutants):

            mutant_idx = num_individual + self.n_elites

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness.values = (np.float64(0.0),)

    # def selEpsilon(self, individuals: list[creator.Individual], k: int, epsilon: float):
    #     selected_individuals = []

    #     best_individuals = tools.selBest(individuals, k)

    #     for i in range(k):
    #         select_best = np.random.choice([True, False], 1, p=[1 - epsilon, epsilon])
    #         if select_best:
    #             selected_individuals.append(best_individuals[i])
    #         else:
    #             selected_individuals.append(random.sample(selected_individuals))

    #     return selected_individuals

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        elites = tools.selBest(subpopulation, self.n_elites)

        non_elites = tools.selTournament(subpopulation, len(subpopulation) - self.n_elites, tournsize=2)

        offspring = elites + non_elites

        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [deepcopy(individual) for individual in offspring]

    def select(self, population):
        # Perform a selection on that subpopulation and add it to the offspring population
        return [self.selectSubPopulation(subpop) for subpop in population]

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def groupTeamsAndEvalInfos(self, teams: list[Team], eval_infos: list[EvalInfo]):

        grouped_teams = []
        grouped_eval_infos = []

        for team_id, _ in enumerate(teams[:: self.n_eval_per_team_set]):

            grouped_teams.append(teams[team_id * self.n_eval_per_team_set : (team_id + 1) * self.n_eval_per_team_set])
            grouped_eval_infos.append(
                eval_infos[team_id * self.n_eval_per_team_set : (team_id + 1) * self.n_eval_per_team_set]
            )

        return grouped_teams, grouped_eval_infos

    def trainFitnessCritics(
        self,
        fitness_critics: list[FitnessCritic],
        eval_infos: list[EvalInfo],
    ):
        fc_loss = []

        # Collect trajectories from eval_infos
        for eval_info in eval_infos:
            for idx in eval_info.team_formation:
                fitness_critics[idx].add(
                    eval_info.joint_traj.observations[:, idx, :], np.float64(eval_info.team_fitness)
                )

        # Train fitness critics
        for fc in fitness_critics:
            accum_loss = fc.train(epochs=self.config["fitness_critic"]["epochs"])
            fc_loss.append(accum_loss)

        return fc_loss

    def assignFitnesses(
        self,
        fitness_critics,
        grouped_teams: list[list[Team]],
        grouped_eval_infos: list[list[EvalInfo]],
    ):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        for teams, eval_infos in zip(grouped_teams, grouped_eval_infos):

            # Aggregate the fitnesses into a big numpy array
            accumulated_fitnesses = [0 for _ in range(self.num_rovers)]

            for eval_info in eval_infos:

                for idx, fit in eval_info.agent_fitnesses:

                    if self.use_fc:
                        accumulated_fitnesses[idx] += fitness_critics[idx].evaluate(
                            eval_info.joint_traj.observations[:, idx, :]
                        )
                    else:
                        accumulated_fitnesses[idx] += fit

            # Group all individuals in teams within this set
            individuals_in_set = {}

            for team in teams:
                for idx, individual in zip(team.combination, team.individuals):
                    individuals_in_set[idx] = individual

            # Put into tuples due to deap.base.Fitness type
            for idx, individual in individuals_in_set.items():
                individual.parameters.fitness.values = (accumulated_fitnesses[idx],)

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, eval_fit_dir):
        header = "generation,team_fitness_aggregated"
        for j in range(self.num_rovers):
            header += ",rover_" + str(j)

        for i in range(self.n_eval_per_team_set):
            header += ",team_fitness_" + str(i)
            for j in range(self.team_size):
                header += ",team_" + str(i) + "_rover_" + str(j)

        header += "\n"
        with open(eval_fit_dir, "w") as file:
            file.write(header)

    def writeEvalFitnessCSV(self, eval_fit_dir, eval_infos):
        gen = str(self.gen)

        # Aggergate the fitnesses into a big numpy array
        num_ind_per_team = self.num_rovers + 1  # +1 to include team fitness
        all_fit = np.zeros(shape=(self.n_eval_per_team_set, num_ind_per_team))

        for n_eval, eval_info in enumerate(eval_infos):

            for idx, fit in eval_info.agent_fitnesses:
                all_fit[n_eval, idx] = fit

            all_fit[n_eval, -1] = eval_info.team_fitness

        # Now compute a sum/average/min/etc dependending on what config specifies
        agg_fit = np.average(all_fit, axis=0)

        # And now record it all, starting with the aggregated one
        agg_team_fit = str(agg_fit[-1])
        agg_agent_fits = [str(fit) for fit in agg_fit[:-1]]
        fit_str = f"{gen},{','.join([agg_team_fit] + agg_agent_fits)},"

        # And now add all the fitnesses from individual trials
        # Each row should have the fitnesses for an evaluation
        for row, eval_info in zip(all_fit, eval_infos):
            team_fit = str(row[-1])
            agent_fits = [str(row[idx]) for idx in eval_info.team_formation]
            fit_str += f"{','.join([team_fit] + agent_fits)},"
        fit_str += "\n"

        # Now save it all to the csv
        with open(eval_fit_dir, "a") as file:
            file.write(fit_str)

    def createFitCritLossCSV(self, trial_dir):
        fit_crit_loss_dir = f"{trial_dir}/fitness_critic_loss.csv"
        header = "generation"

        for j in range(self.num_rovers):
            header += ",loss_" + str(j)

        header += "\n"

        with open(fit_crit_loss_dir, "w") as file:
            file.write(header)

    def writeFitCritLossCSV(self, trial_dir, fitness_critics_loss):
        fit_crit_loss_dir = f"{trial_dir}/fitness_critic_loss.csv"
        gen = str(self.gen)

        fit_str = f"{gen},"

        for loss in fitness_critics_loss:
            fit_str += f"{str(loss)},"

        fit_str += "\n"

        # Now save it all to the csv
        with open(fit_crit_loss_dir, "a") as file:
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
                joint_traj = eval_info.joint_traj

                # We're going to pad the actions with None because
                # the agents cannot take actions at the last timestep, but
                # there is a final joint state/observations
                action_padding = []
                for action in joint_traj.actions[0]:
                    action_padding.append([None for _ in action])

                joint_traj.actions.append(action_padding)

                for agent_positions, poi_positions, joint_observation, joint_action in zip(
                    joint_traj.states.agent_positions,
                    joint_traj.states.poi_positions,
                    joint_traj.observations,
                    joint_traj.actions,
                ):
                    # Aggregate state info
                    state_list = []

                    for pos in agent_positions:
                        state_list += [str(coord) for coord in pos]
                        state_str = ",".join(state_list)

                    for pos in poi_positions:
                        state_list += [str(coord) for coord in pos]
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

    def init_fitness_critics(self):
        # Initialize fitness critics
        fc = None

        if self.use_fc:

            loss_fn = 0

            match self.fit_crit_loss_type:
                case "MSE":
                    loss_fn = 0
                case "MAE":
                    loss_fn = 1
                case "MSE+MAE":
                    loss_fn = 2

            fc = [
                FitnessCritic(
                    device=DEVICE,
                    model_type=self.fit_crit_type,
                    loss_fn=loss_fn,
                    episode_size=self.n_steps,
                    hidden_size=self.fit_crit_num_hidden,
                    n_layers=len(self.fit_crit_num_hidden),
                )
                for _ in range(self.num_rovers)
            ]

        return fc

    def run(self):

        # Set trial directory name
        trial_folder_name = "_".join(("trial", str(self.trial_id), self.trial_name))
        trial_dir = os.path.join(self.trials_dir, trial_folder_name)
        eval_fit_dir = f"{trial_dir}/fitness.csv"
        checkpoint_name = os.path.join(trial_dir, "checkpoint.pickle")

        # Create directory for saving data
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Load checkpoint
        if self.load_checkpoint:

            with open(checkpoint_name, "rb") as handle:
                checkpoint = pickle.load(handle)
                pop = checkpoint["population"]
                self.checkpoint_gen = checkpoint["gen"]
                fc_params = checkpoint["fitness_critics"]
            
            #Load fitness critics params
            if self.use_fc:
                fitness_critics = self.init_fitness_critics()
                for fc, params in zip(fitness_critics, fc_params):
                    fc.model.set_params(params)
            else: 
                fitness_critics = None
            
            #Set fitness csv file to checkpoint
            new_fit_path = os.path.join(trial_dir, "fitness_edit.csv")
            with open(eval_fit_dir, 'r') as inp, open(new_fit_path, 'w') as out:
                writer = csv.writer(out)
                for row in csv.reader(inp):
                    if row[0].isdigit():
                        gen = int(row[0])
                        if gen <= self.checkpoint_gen:
                            writer.writerow(row)
                    else:
                        writer.writerow(row)

            #Remove old fitness file
            os.remove(eval_fit_dir)
            #Rename new fitness file
            os.rename(new_fit_path, eval_fit_dir)

        else:
            # Initialize the population
            pop = self.toolbox.population()

            # Create csv file for saving evaluation fitnesses
            self.createEvalFitnessCSV(eval_fit_dir)

            # Create csv file for saving fitness critic losses
            self.createFitCritLossCSV(trial_dir)

            #Initialize fitness critics
            if self.use_fc:
                fitness_critics = self.init_fitness_critics()
            else: 
                fitness_critics = None


        for n_gen in range(self.n_gens + 1):

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

            # Train Fitness Critics
            if self.use_fc:
                fc_loss = self.trainFitnessCritics(fitness_critics, eval_infos)
                self.writeFitCritLossCSV(trial_dir, fc_loss)

            # Regroup sets of teams with their respective sets of eval_infos
            grouped_teams, grouped_eval_infos = self.groupTeamsAndEvalInfos(teams, eval_infos)

            # Now assign fitnesses to each individual
            self.assignFitnesses(fitness_critics, grouped_teams, grouped_eval_infos)

            # Evaluate a team with the best individual from each subpopulation
            eval_infos = self.evaluateBestTeam(offspring)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save fitnesses
            self.writeEvalFitnessCSV(eval_fit_dir, eval_infos)

            # Save trajectories and checkpoint
            if n_gen % self.num_gens_between_save == 0:

                # Save trajectories
                self.writeEvalTrajs(trial_dir, eval_infos)

                # Save checkpoint
                with open(os.path.join(trial_dir, "checkpoint.pickle"), "wb") as handle:
                    pickle.dump(
                        {
                            "population": pop,
                            "gen": n_gen,
                            "fitness_critics": (
                                [fc.params for fc in fitness_critics] if self.use_fc else None
                            ),
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        if self.use_multiprocessing:
            self.pool.close()


def runCCEA(config_dir, experiment_name: str, trial_id: int, load_checkpoint: bool):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir, experiment_name, trial_id, load_checkpoint)
    return ccea.run()
