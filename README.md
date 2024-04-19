# Trajectory Recovery

This repository contains an implementation of the trajectory recovery algorithm proposed by Xu et al. 
It is supplied as a Python module and as a Jupyter Notebook with detailed step-by-step explanations.

We evaluate the algorithm on two open-source datasets: GeoLife and Porto Taxi.

- `data/` contains all the preprocessed datasets.
- `results/` contains plotted results and pickle files of the results. It also contains plots of all the predicted trajectories against the true ones.
- `algorithm_walkthrough.ipynb` is the Jupyter Notebook explaining the algorithm.
- `trajectory_recovery.py` is the Python module.

Reference:
- F. Xu, Z. Tu, Y. Li, P. Zhang, X. Fu, and D. Jin, “Trajectory recovery from ash: User privacy is not preserved in aggregated mobility data,” in Proceedings of the 26th International Conference on World Wide Web, ser. WWW ’17. International World Wide Web Conferences Steering Committee, Apr. 2017. [Online]. Available: http://dx.doi.org/10.1145/3038912.3052620
