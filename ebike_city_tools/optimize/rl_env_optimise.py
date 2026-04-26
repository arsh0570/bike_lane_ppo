import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import networkx as nx
from ebike_city_tools.metrics import compute_travel_times_in_graph
from ebike_city_tools.iterative_algorithms import transform_car_to_bike_edge
from ebike_city_tools.utils import compute_car_time, compute_edgedependent_bike_time

class BikeNetworkEnv(gym.Env):
    def __init__(self, G_lane, od, max_steps=20, shared_lane_factor=2):
        super(BikeNetworkEnv, self).__init__()
        self.G_initial = G_lane.copy()
        self.shared_lane_factor = shared_lane_factor
        
        # --- FIX: Ensure bike_time and car_time exist on edges ---
        for u, v, k, data in self.G_initial.edges(data=True, keys=True):
            self.G_initial[u][v][k]["car_time"] = compute_car_time(data)
            self.G_initial[u][v][k]["bike_time"] = compute_edgedependent_bike_time(
                data, shared_lane_factor=self.shared_lane_factor
            )
        
        self.G_current = self.G_initial.copy()
        self.od = od
        self.max_steps = max_steps
        self.current_step = 0
        
        self.edges = list(self.G_initial.edges(keys=True))
        self.action_space = spaces.Discrete(len(self.edges))
        
        # --- FIX: Flatten observation space (len(edges) * 3 features) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.edges) * 3,), dtype=np.float32
        )

    def _get_obs(self):
        obs = []
        for u, v, k in self.edges:
            data = self.G_current[u][v][k]
            is_bike = 1.0 if data.get("lanetype") == "P" else 0.0
            obs.append([data['distance'], data['gradient'], is_bike])
        # --- FIX: Return a flattened 1D array ---
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        edge_to_convert = self.edges[action]
        u, v, k = edge_to_convert
        
        reward = 0
        terminated = False
        
        if self.G_current[u][v][k].get("lanetype") == "P":
            reward = -5  # Small penalty for redundant action
        else:
            temp_G = self.G_current.copy()
            # This call should now find 'bike_time' successfully!
            transform_car_to_bike_edge(temp_G, edge_to_convert, self.shared_lane_factor)
            
            # Simple connectivity check
            # Note the addition of keys=True
            car_graph = nx.MultiDiGraph([(u, v, d) for u, v, k, d in temp_G.edges(data=True, keys=True) if d.get("lanetype") != "P"])
            if nx.is_strongly_connected(car_graph):
                self.G_current = temp_G
                # Added True (or False) for the weight_od_flow argument
                b_time, c_time = compute_travel_times_in_graph(self.G_current, self.od, "od", True)
                # Balanced reward: focus on bike time reduction
                reward = (100.0 / (b_time + 1e-6)) - (0.7 * c_time)
            else:
                reward = -20 # Higher penalty for disconnecting the city

        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            
        return self._get_obs(), float(reward), terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.G_current = self.G_initial.copy()
        self.current_step = 0
        return self._get_obs(), {}