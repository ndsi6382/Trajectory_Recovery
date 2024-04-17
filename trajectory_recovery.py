import math
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


class TrajectoryRecovery():
    def __init__(
        self,
        aggregated_dataset: pd.DataFrame,
        grid: dict,
        num_trajectories: int,
        num_locations: int,
        num_timesteps: int,
        num_intervals_per_day: int
    ):
        self.agg = aggregated_dataset
        self.truth = None
        self.grid = grid
        self.N = num_trajectories
        self.M = num_locations
        self.T = num_timesteps
        self.D = num_intervals_per_day
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


    def run_algorithm(self, verbose=False):
        self.__night__(verbose)
        self.__day__(verbose)
        self.__across__(verbose)


    def __night__(self, verbose=False):
        if verbose:
            print("Starting nighttime trajectory recovery process.")

        cost = np.zeros((self.N, self.N))
        for i in [x for x in range(self.T) if x % self.D < self.D // 4 and x + 1 < self.T]:
            if verbose and i % self.D == 0:
                print(f"Night {i // self.D} processing...")
            for u in range(self.N):
                for l in range(self.N):
                    loc_i = self.grid[self.pred[u][i]]
                    loc_j = self.grid[self.locs[i+1][l]]
                    cost[u][l] = math.dist(loc_i, loc_j)
            row_assn, col_assn = scipy.optimize.linear_sum_assignment(cost, maximize=False)
            for u, l in enumerate(col_assn):
                self.pred[u][i+1] = self.locs[i+1][l]
        if verbose:
            TrajectoryRecovery.preview_matrix(self.pred, 10, self.D // 4)
            print()


    def __day__(self, verbose=False):
        if verbose:
            print("Starting daytime trajectory recovery process.")

        cost = np.zeros((self.N, self.N))
        for i in [x for x in range(self.T) if x % self.D >= self.D // 4 and x + 1 < self.T]:
            if verbose and i % self.D == self.D // 4:
                print(f"Day {i // self.D} processing...")
            for u in range(self.N):
                for l in range(self.N):
                    q_t = self.grid[self.pred[u][i]]
                    q_t1 = self.grid[self.pred[u][i-1]]
                    loc_i = (q_t[0]+(q_t[0]-q_t1[0]), q_t[1]+(q_t[1]-q_t1[1]))
                    loc_j = self.grid[self.locs[i+1][l]]
                    cost[u][l] = math.dist(loc_i, loc_j)
            row_assn, col_assn = scipy.optimize.linear_sum_assignment(cost, maximize=False)
            for u, l in enumerate(col_assn):
                self.pred[u][i+1] = self.locs[i+1][l]
        if verbose:
            TrajectoryRecovery.preview_matrix(self.pred, 10, self.D)
            print()


    def __across__(self, verbose=False):
        if verbose:
            print("Starting subtrajectory linking process.")

        for i in range(self.N):
            self.pred[i] = [self.pred[i][j:j+self.D] for j in range(0, self.T, self.D)]

        days = [[x[i] for x in self.pred] for i in range(math.ceil(self.T / self.D))]
        links = [None for _ in range(len(days)-1)]
        cost = np.zeros((self.N, self.N))
        for i in range(len(days)-1):
            if verbose:
                print(f"Day {i} processing...")
            for a in range(self.N):
                for b in range(self.N):
                    cost[a][b] = TrajectoryRecovery.gain(days[i][a], days[i+1][b])
            row_assn, col_assn = scipy.optimize.linear_sum_assignment(cost, maximize=False)
            links[i] = col_assn

        self.pred = [days[0][i] for i in range(self.N)]
        for i in range(self.N):
            curr_row = i
            for j in range(len(days)-1):
                next_row = links[j][curr_row]
                self.pred[i] = self.pred[i] + days[j+1][next_row]
                curr_row = next_row
        if verbose:
            TrajectoryRecovery.preview_matrix(self.pred, 10, self.T)
            print()


    def evaluate(self, truth_dataset: list[list], verbose=False):
        if verbose:
            print("Starting evaluation process.")
        self.truth = truth_dataset
        self.compare = []
        self.result = {
            'accuracy': 0,
            'recovery_error': 0,
            'uniqueness': {
                'predicted': dict(),
                'truth': dict(),
            }
        }

        accuracy_matrix = np.zeros((self.N,self.N))
        error_matrix = np.zeros((self.N,self.N))
        for i in range(self.N): # predicted trajectories
            print(f"Evaluating trajectory {i}...")
            for j in range(self.N): # truth trajectories
                error = 0
                acc = 0
                for k in range(self.T): # timesteps
                    pred_loc = self.grid[self.pred[i][k]]
                    x = math.dist(pred_loc, self.truth[j][k])
                    error += x # location error
                    if x == 0: # accuracy
                        acc += 1
                error_matrix[i][j] = error
                accuracy_matrix[i][j] = acc / self.T

        #compare[i] = j means the i-th predicted trajectory matches the j-th true trajectory.
        _, self.compare = scipy.optimize.linear_sum_assignment(error_matrix, maximize=False)

        for i, j in enumerate(self.compare):
            self.result['accuracy'] += accuracy_matrix[i][j] / self.N
            self.result['recovery_error'] += error_matrix[i][j]

        for i in range(1,6):
            self.result["uniqueness"]["predicted"][i] = TrajectoryRecovery.uniqueness(self.pred, i)
            self.result["uniqueness"]["truth"][i] = TrajectoryRecovery.uniqueness(self.truth, i)

        if verbose:
            print()
            for k, v in self.result.items():
                print(f"{k}: {v}")
            print(self.compare)
            print()


    def gain(trajectory_1, trajectory_2):
        return TrajectoryRecovery.__entropy__(trajectory_1 + trajectory_2) - ((
                       TrajectoryRecovery.__entropy__(trajectory_1) +
                       TrajectoryRecovery.__entropy__(trajectory_2)
                   ) / 2
               )


    def __entropy__(trajectory):
        freq = [0 for _ in range(max(trajectory)+1)]
        for x in trajectory:
            freq[x] += 1
        result = 0
        for f in [x for x in freq if x > 0]:
            result -= (f/len(trajectory))*math.log2(f/len(trajectory))
        return result


    def visualise(self, num_timestamps=None):
        if not self.result:
            print("Results have not been evaluated. Call evaluate() to generate results.")
            return

        plots = []
        if not num_timestamps:
          num_timestamps = self.D

        for i in range(self.N):
            pred_traj = self.pred[i][:num_timestamps]
            pred_traj = [self.grid[x] for x in pred_traj]
            true_traj = self.truth[self.compare[i]][:num_timestamps]
            x1, y1 = zip(*pred_traj)
            x2, y2 = zip(*true_traj)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].plot(x1, y1, color='red')
            axs[0].set_title(f'Predicted Trajectory (No. {i})')
            axs[0].set_xlabel('Longitude')
            axs[0].set_ylabel('Lattitude')
            axs[0].grid()
            axs[1].plot(x2, y2, color='blue')
            axs[1].set_title(f'True Trajectory (No. {self.compare[i]})')
            axs[1].set_xlabel('Longitude')
            axs[1].set_ylabel('Lattitude')
            axs[1].grid()
            fig.tight_layout()
            plots.append(fig)

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


    def preview_matrix(matrix, rows, cols):
        for i in range(rows):
            print(matrix[i][:cols])


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
