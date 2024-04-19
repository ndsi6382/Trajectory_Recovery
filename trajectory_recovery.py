import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from geopy.distance import distance as lat_lon_distance
from tqdm import tqdm


class TrajectoryRecovery():
    def __init__(
        self,
        aggregated_dataset: pd.DataFrame,
        grid: dict,
        num_trajectories: int,
        num_locations: int,
        num_timesteps: int,
        num_intervals_per_day: int,
        cartesian: bool = True
    ):
        self.agg = aggregated_dataset
        self.truth = None
        self.grid = grid
        self.N = num_trajectories
        self.M = num_locations
        self.T = num_timesteps
        self.D = num_intervals_per_day
        if cartesian:
            self.dist_fn = math.dist
        else:
            self.dist_fn = lat_lon_distance

        self.locs = [dict() for _ in range(self.T)]
        self.pred = [[-1]*self.T for _ in range(self.N)]
        self.result = None

        for i in range(self.T):
            row = aggregated_dataset.iloc[[i]].values[0]
            tmp = 0
            for j, val in enumerate(row):
                for k in range(int(val)):
                    self.locs[i][tmp+k] = j
                    if i % self.D == 0:
                        self.pred[tmp+k][i] = j
                tmp += int(val)


    def run_algorithm(self, check: bool = False):
        self.__night__()
        self.__day__()
        self.__across__()
        if check:
            for traj in self.pred:
                assert all(x != -1 for x in traj)


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
            for u, l in enumerate(col_assn):
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
            for u, l in enumerate(col_assn):
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


    def evaluate(self, truth_dataset: list[list]):
        self.truth = truth_dataset
        self.result = {
            'accuracy': 0,
            'recovery_error': 0,
            'uniqueness': {
                'predicted': dict(),
                'truth': dict(),
            },
            'compare': list(),
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

        _, compare = linear_sum_assignment(error_matrix, maximize=False)
        for i, j in enumerate(compare):
            self.result['accuracy'] += accuracy_matrix[i][j] / self.N
            self.result['recovery_error'] += error_matrix[i][j]
            self.result['compare'].append((i,j))
        for i in range(1,6):
            self.result["uniqueness"]["predicted"][i] = TrajectoryRecovery.uniqueness(self.pred, i)
            self.result["uniqueness"]["truth"][i] = TrajectoryRecovery.uniqueness(self.truth, i)


    def gain(trajectory_1: list[int], trajectory_2: list[int]):
        return TrajectoryRecovery.__entropy__(trajectory_1 + trajectory_2) - ((
                       TrajectoryRecovery.__entropy__(trajectory_1) +
                       TrajectoryRecovery.__entropy__(trajectory_2)
                   ) / 2
               )


    def __entropy__(trajectory: list[int]):
        freq = [0 for _ in range(max(trajectory)+1)]
        for x in trajectory:
            freq[x] += 1
        result = 0
        for f in [x for x in freq if x > 0]:
            result -= (f/len(trajectory))*math.log2(f/len(trajectory))
        return result


    def visualise(self, timestep_range: tuple[int,int] = None):
        if not self.result:
            print("Results have not been evaluated. Call evaluate() to generate results.")
            return
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
            axs[0].set_ylabel('Lattitude')
            axs[0].grid()
            axs[1].plot(x2, y2, color='blue')
            axs[1].set_title(f'True Trajectory (No. {j})')
            axs[1].set_xlabel('Longitude')
            axs[1].set_ylabel('Lattitude')
            axs[1].grid()
            fig.tight_layout()
            plots.append(fig)
        plt.rcParams['figure.max_open_warning'] = True
        return plots


    def uniqueness(data, k):
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


    def get_predictions(self):
        if not self.pred:
            print("Algorithm has not been run. Call run_algorithm() to generate predictions.")
        else:
            return self.pred


    def get_results(self):
        if not self.result:
            print("Results have not been evaluated. Call evaluate() to generate results.")
        else:
            return self.result
