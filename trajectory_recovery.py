import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from geopy.distance import distance as lat_lon_distance
from tqdm import tqdm
import copy
from typing import *
from Levenshtein import distance as lev

class TrajectoryRecoveryA():
    def __init__(
        self,
        aggregated_dataset: Union[pd.DataFrame, np.ndarray],
        grid: Union[dict, list, np.ndarray],
        num_trajectories: int,
        num_locations: int,
        num_timesteps: int,
        num_timesteps_per_day: int,
        cartesian: bool = True
    ):
        if type(aggregated_dataset == pd.DataFrame):
            self.agg = aggregated_dataset.values
        else:
            self.agg = aggregated_dataset
        self.truth = None
        self.grid = grid
        self.N = num_trajectories
        self.M = num_locations
        self.T = num_timesteps
        self.D = num_timesteps_per_day
        if cartesian:
            self.dist_fn = math.dist
        else:
            self.dist_fn = lat_lon_distance

        self.locs = [dict() for _ in range(self.T)]
        self.pred = [[-1]*self.T for _ in range(self.N)]
        self.result = None

        for i in range(self.T):
            row = self.agg[i]
            tmp = 0
            for j, val in enumerate(row):
                for k in range(val):
                    self.locs[i][tmp+k] = j
                    if i % self.D == 0:
                        self.pred[tmp+k][i] = j
                tmp += val


    def run_algorithm(self):
        self.__night__()
        self.__day__()
        self.__across__()


    def __night__(self):
        pbar = tqdm(total=((self.T // self.D) + (self.T % self.D > 1)),
                    desc="Recovering Night Trajectories", unit="night", ncols=100)
        cost = np.zeros((self.N, self.N))
        for i in [x for x in range(self.T) if x % self.D < self.D // 4 and x < self.T - 1]:
            for u in range(self.N):
                for l in range(self.N):
                    loc_i = self.grid[self.pred[u][i]]
                    loc_j = self.grid[self.locs[i+1][l]]
                    cost[u][l] = self.dist_fn(loc_i, loc_j)
            row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
            for u, l in zip(row_assn, col_assn):
                self.pred[u][i+1] = self.locs[i+1][l]
            if i % self.D == 0:
                pbar.update()
        pbar.close()


    def __day__(self):
        pbar = tqdm(total=((self.T // self.D) + (self.T % self.D > (self.D // 4 + 1))),
                    desc="Recovering Day Trajectories", unit="day", ncols=100)
        cost = np.zeros((self.N, self.N))
        for i in [x for x in range(self.T) if self.D // 4 <= x % self.D < self.D - 1 and x < self.T - 1]:
            for u in range(self.N):
                for l in range(self.N):
                    q_t = self.grid[self.pred[u][i]]
                    q_t1 = self.grid[self.pred[u][i-1]]
                    loc_i = (q_t[0]+(q_t[0]-q_t1[0]), q_t[1]+(q_t[1]-q_t1[1]))
                    loc_j = self.grid[self.locs[i+1][l]]
                    cost[u][l] = self.dist_fn(loc_i, loc_j)
            row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
            for u, l in zip(row_assn, col_assn):
                self.pred[u][i+1] = self.locs[i+1][l]
            if i % self.D == self.D // 4:
                pbar.update()
        pbar.close()


    def __across__(self):
        for i in range(self.N):
            self.pred[i] = [self.pred[i][j:j+self.D] for j in range(0, self.T, self.D)]
        days = [[x[i] for x in self.pred] for i in range(math.ceil(self.T / self.D))]
        links = [None for _ in range(len(days)-1)]

        pbar = tqdm(total=len(days)-1, desc="Linking Sub-trajectories", unit="day", ncols=100)
        cost = np.zeros((self.N, self.N))
        for i in range(len(days)-1):
            for a in range(self.N):
                for b in range(self.N):
                    cost[a][b] = TrajectoryRecovery.gain(days[i][a], days[i+1][b])
            row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
            links[i] = col_assn
            pbar.update()
        pbar.close()

        self.pred = [days[0][i] for i in range(self.N)]
        for i in range(self.N):
            curr_row = i
            for j in range(len(days)-1):
                next_row = links[j][curr_row]
                self.pred[i] = self.pred[i] + days[j+1][next_row]
                curr_row = next_row


    def evaluate(self, truth_dataset: list[list[tuple]]):
        self.truth = truth_dataset
        self.result = {
            'accuracy': 0,
            'recovery_error': 0,
            'uniqueness': {
                'predicted': dict(),
                'truth': dict(),
            },
            'compare': list(),
            'levenshtein': 0,
        }
        pbar = tqdm(total=self.N, desc="Evaluating Trajectories", unit="traj.", ncols=100)

        accuracy_matrix = np.zeros((self.N, self.N))
        error_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                error = 0
                acc = 0
                for k in range(self.T):
                    pred_loc = self.grid[self.pred[i][k]]
                    x = self.dist_fn(pred_loc, self.truth[j][k])
                    error += x
                    if x == 0:
                        acc += 1
                error_matrix[i][j] = error
                accuracy_matrix[i][j] = acc / self.T
            pbar.update()
        pbar.close()

        pred_traj, true_traj = linear_sum_assignment(error_matrix, maximize=False)
        for i, j in zip(pred_traj, true_traj):
            self.result['accuracy'] += accuracy_matrix[i][j] / self.N
            self.result['recovery_error'] += error_matrix[i][j]
            self.result['compare'].append((i,j))
            self.result['levenshtein'] += (lev([self.grid[x] for x in self.pred[i]], self.truth[j]) / self.T) / self.N
        for i in range(1,6):
            self.result["uniqueness"]["predicted"][i] = TrajectoryRecovery.uniqueness(self.pred, i)
            self.result["uniqueness"]["truth"][i] = TrajectoryRecovery.uniqueness(self.truth, i)


    def gain(trajectory_1: list, trajectory_2: list):
        return TrajectoryRecovery.__entropy__(trajectory_1 + trajectory_2) - ((
                       TrajectoryRecovery.__entropy__(trajectory_1) +
                       TrajectoryRecovery.__entropy__(trajectory_2)
                   ) / 2
               )


    def __entropy__(trajectory: list):
        freq = {x:0 for x in trajectory}
        for x in trajectory:
            freq[x] += 1
        result = 0
        for f in [v for k, v in freq.items() if v > 0]:
            result -= (f/len(trajectory))*math.log2(f/len(trajectory))
        return result


    def visualise(self, timestep_range: tuple[int,int] = None):
        if not self.result:
            raise RuntimeError("Results have not been evaluated. Call evaluate() to generate results.")
        plots = []
        if not timestep_range:
            timestep_range = (0, self.D)
        plt.rcParams['figure.max_open_warning'] = False
        for (i, j) in self.result['compare']:
            pred_traj = self.pred[i][timestep_range[0]:timestep_range[1]]
            pred_traj = [self.grid[x] for x in pred_traj]
            true_traj = self.truth[j][timestep_range[0]:timestep_range[1]]
            x1, y1 = zip(*pred_traj)
            x2, y2 = zip(*true_traj)
            fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            axs[0].plot(x1, y1, color='red')
            axs[0].set_title(f'Predicted Trajectory (No. {i})')
            axs[0].set_xlabel('Longitude')
            axs[0].set_ylabel('Latitude')
            axs[0].grid()
            axs[1].plot(x2, y2, color='blue')
            axs[1].set_title(f'True Trajectory (No. {j})')
            axs[1].set_xlabel('Longitude')
            axs[1].grid()
            axs[0].tick_params(labelbottom=False, labelleft=False)
            axs[1].tick_params(labelbottom=False, labelleft=False)
            fig.tight_layout()
            plots.append(fig)
        plt.rcParams['figure.max_open_warning'] = True
        return plots


    def uniqueness(data: list[list], k: int):
        top_k_locs = [set() for _ in range(len(data))]
        for u in range(len(data)):
            freq = dict()
            for l in range(len(data[0])):
                if data[u][l] not in freq.keys():
                    freq[data[u][l]] = 1
                else:
                    freq[data[u][l]] += 1
            freq = sorted([(k, v) for k, v in freq.items()], key=lambda x: x[1], reverse=True)
            top_k_locs[u] = set([x[0] for x in freq][:k])
        return len(set(map(frozenset, top_k_locs))) / len(data)


    def get_predictions(self, location_type: Literal['coordinate', 'id'] = 'coordinate'):
        if not self.pred:
            raise RuntimeError("Algorithm has not been run. Call run_algorithm() to generate predictions.")
        else:
            if location_type == "coordinate":
                return [[self.grid[l] for l in u] for u in self.pred] #return self.pred
            elif location_type == "id":
                return self.pred
            else:
                raise ValueError("Invalid location_type given. Must be 'coordinate' or 'id'.")


    def get_results(self):
        if not self.result:
            raise RuntimeError("Results have not been evaluated. Call evaluate() to generate results.")
        else:
            return self.result


class TrajectoryRecoveryB(TrajectoryRecoveryA):
    """
    Notes:
    To use the 'stages' of the original paper, the correct steps for night/day are:
        self.__night__([x for x in range(len(self.curr_day[0])-1) if x < self.D // 4], d) # midnight to 6am
        self.__day__([x for x in range(len(self.curr_day[0])-1) if x >= self.D // 4], d) # 6am onwards

    The inner block in the __day__() function may alternatively written as:
        for l in range(self.N):
            q_t = self.grid[self.curr_day[u][i]]
            q_t1 = self.grid[self.curr_day[u][i-1]]
            loc_i = (q_t[0]+(q_t[0]-q_t1[0]), q_t[1]+(q_t[1]-q_t1[1])) # estimated location
            loc_j = self.grid[self.locs[day*self.D + i+1][l]] # candidate location
            cost[u][l] = self.dist_fn(loc_i, loc_j)
            for h in histories:
                cost[u][l] = min(cost[u][l], self.dist_fn(self.grid[h], loc_j))
    """

    def __init__(
        self,
        aggregated_dataset: Union[pd.DataFrame, np.ndarray],
        grid: Union[dict, list, np.ndarray],
        num_trajectories: int,
        num_locations: int,
        num_timesteps: int,
        num_timesteps_per_day: int,
        cartesian: bool = True
    ):
        if type(aggregated_dataset == pd.DataFrame):
            self.agg = aggregated_dataset.values
        else:
            self.agg = aggregated_dataset
        self.truth = None
        self.grid = grid
        self.N = num_trajectories
        self.M = num_locations
        self.T = num_timesteps
        self.D = num_timesteps_per_day
        if cartesian:
            self.dist_fn = math.dist
        else:
            self.dist_fn = lat_lon_distance
        self.locs = [dict() for _ in range(self.T)] # locs[time][col] = loc_id
        self.revlocs = [dict() for _ in range(self.T)] # revlocs[time][loc_id] = list_of_columns
        self.pred = [[] for _ in range(self.N)]
        self.bigrams = np.zeros((self.M, self.M)) # locations must be stored by ID, not tuple.
        self.bigrams_sums = np.zeros(self.M) # tracks the max. frequency of ANY destination, for each source location
        self.result = None

        for i in range(self.T):
            row = self.agg[i]
            tmp = 0
            for j, val in enumerate(row):
                for k in range(val):
                    self.locs[i][tmp+k] = j
                    if j not in self.revlocs[i].keys():
                        self.revlocs[i][j] = [tmp+k]
                    else:
                        self.revlocs[i][j].append(tmp+k)
                tmp += val


    def run_algorithm(self, lookback: int = 1):
        lookback = max(1, lookback)
        pbar = tqdm(total=math.ceil(self.T / self.D), desc="Recovering Trajectories", unit="day", ncols=100)
        for d in range(math.ceil(self.T / self.D)): # day
            if (d+1)*self.D > self.T:
                self.curr_day = [[-1] * (self.T - d*self.D) for _ in range(self.N)]
            else:
                self.curr_day = [[-1]*self.D for _ in range(self.N)]
            row = self.agg[0]
            tmp = 0
            for j, val in enumerate(row):
                for k in range(val):
                    self.curr_day[tmp+k][0] = j
                tmp += val
            self.__night__([x for x in range(len(self.curr_day[0])-1) if x == 0], d) # midnight only
            self.__day__([x for x in range(len(self.curr_day[0])-1) if x > 0], d) # all timesteps except the first
            if d == 0:
                self.pred = copy.deepcopy(self.curr_day)
            else:
                self.__across__(d, lookback)
            for u in range(self.N): # Update Bigrams from self.pred
                for i in range(max(1, d*self.D), min((d+1)*self.D, self.T)):
                    self.bigrams[self.pred[u][i-1]][self.pred[u][i]] += 1
                    self.bigrams_sums[self.pred[u][i-1]] = max(self.bigrams_sums[self.pred[u][i-1]], self.bigrams[self.pred[u][i-1]][self.pred[u][i]])
            pbar.update()
        pbar.close()
        del self.curr_day


    def __night__(self, steps, day):
        cost = np.zeros((self.N, self.N))
        for i in steps:
            for u in range(self.N):
                for l in range(self.N):
                    loc_i = self.grid[self.curr_day[u][i]]
                    loc_j = self.grid[self.locs[day*self.D + i+1][l]]
                    cost[u][l] = self.dist_fn(loc_i, loc_j)
            row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
            for u, l in zip(row_assn, col_assn):
                self.curr_day[u][i+1] = self.locs[day*self.D + i+1][l]


    def __day__(self, steps, day):
        cost = np.zeros((self.N, self.N))
        for i in steps:
            for u in range(self.N):
                histories = [j for j, x in enumerate(self.bigrams[self.curr_day[u][i]]) if x == self.bigrams_sums[self.curr_day[u][i]] and x > 0]
                for loc_id, cols in self.revlocs[day*self.D + i+1].items():
                    q_t = self.grid[self.curr_day[u][i]]
                    q_t1 = self.grid[self.curr_day[u][i-1]]
                    loc_i = (q_t[0]+(q_t[0]-q_t1[0]), q_t[1]+(q_t[1]-q_t1[1])) # estimated location
                    loc_j = self.grid[loc_id] # candidate location
                    loc_cost = self.dist_fn(loc_i, loc_j)
                    for h in histories:
                        loc_cost = min(loc_cost, self.dist_fn(self.grid[h], loc_j))
                    for c in cols:
                        cost[u][c] = loc_cost
            row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
            for u, l in zip(row_assn, col_assn):
                self.curr_day[u][i+1] = self.locs[day*self.D + i+1][l]


    def __across__(self, day, lookback):
        cost = np.zeros((self.N, self.N))
        for a in range(self.N): # pred
            for b in range(self.N): # curr
                min_cost = float('inf')
                for i in range(lookback):
                    min_cost = min(TrajectoryRecovery.gain(self.pred[a][max(0,day-i-1)*self.D:max(1,day-i)*self.D], self.curr_day[b]), min_cost)
                cost[a][b] = min_cost # cost that pred[a] matches curr_day[b]
        row_assn, col_assn = linear_sum_assignment(cost, maximize=False)
        for p, c in zip(row_assn, col_assn):
            self.pred[p] += self.curr_day[c]


TrajectoryRecovery = TrajectoryRecoveryB # Alias the enhanced version