import time
import os
import pandas as pd
import networkx as nx
import numpy as np
import torch

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import (
    compute_car_time,
    compute_edgedependent_bike_time,
    output_to_dataframe,
    fix_multilane_bike_lanes,
)
from ebike_city_tools.graph_utils import lane_to_street_graph
from ebike_city_tools.iterative_algorithms import transform_car_to_bike_edge
from ebike_city_tools.metrics import compute_travel_times_in_graph

try:
    from stable_baselines3 import PPO
except ImportError:
    print("RL features require stable-baselines3. Install via: pip install stable-baselines3")

FLOW_CONSTANT = 1


class ParetoRoundOptimize:
    def __init__(self, G_lane, od, sp_method="od", optimize_every_x=5, **kwargs):
        """
        Modified ParetoRoundOptimize to support both LP and RL-based rounding.
        """
        self.G_lane = G_lane
        assert nx.is_strongly_connected(G_lane), "Lane graph is not strongly connected"
        self.od = od
        self.sp_method = sp_method
        self.optimize_every_x = optimize_every_x
        self.optimize_kwargs = kwargs
        self.shared_lane_factor = self.optimize_kwargs.get("shared_lane_factor", 2)

        # transform to street graph
        self.G_street = lane_to_street_graph(G_lane)
        self.runtimes = {"time_init": [], "time_optim": []}

        # init variables for the pareto frontier
        self.reset_pareto_variables()

    def reset_pareto_variables(self):
        self.pareto_df = []
        self.modified_G_lane = self.G_lane.copy()
        nx.set_edge_attributes(self.modified_G_lane, "M>", name="lanetype")
        
        car_time, bike_time = {}, {}
        for u, v, k, data in self.modified_G_lane.edges(data=True, keys=True):
            e = (u, v, k)
            car_time[e] = compute_car_time(data)
            bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=self.shared_lane_factor)
        
        nx.set_edge_attributes(self.modified_G_lane, car_time, name="car_time")
        nx.set_edge_attributes(self.modified_G_lane, bike_time, name="bike_time")

        self.car_graph = self.G_lane.copy()
        self.is_bike = {edge: False for edge in self.G_lane.edges(keys=False)}
        self.fixed_capacities = pd.DataFrame(columns=["Edge", "u_b(e)", "u_c(e)", "capacity"])
        self.total_capacities = nx.get_edge_attributes(self.G_street, "capacity")

    def optimize(self, fixed_capacities):
        """Standard LP optimization call (The 'Slow' way)"""
        obj_value = None
        ip = None
        counter = 0
        while obj_value is None:
            if counter >= 1:
                old_valid_edges = self.optimize_kwargs.get("valid_edges_k", 0)
                self.optimize_kwargs["valid_edges_k"] = old_valid_edges * 2
            
            tic = time.time()
            ip = define_IP(self.G_street, od_df=self.od, fixed_edges=fixed_capacities, **self.optimize_kwargs)
            toc = time.time()
            ip.verbose = False
            ip.optimize()
            obj_value = ip.objective_value
            toc_optim = time.time()
            counter += 1

        self.runtimes["time_init"].append(toc - tic)
        self.runtimes["time_optim"].append(toc_optim - toc)
        return output_to_dataframe(ip, self.G_street, fixed_edges=fixed_capacities)

    def allocate_bike_edge(self, edge_to_transform, assert_greater_0=False, remove_from_car=False):
        """Helper to transform a car lane into a bike lane and update fixed capacities"""
        self.is_bike[edge_to_transform[:2]] = True
        new_edge = transform_car_to_bike_edge(self.modified_G_lane, edge_to_transform, self.shared_lane_factor)
        self.is_bike[new_edge[:2]] = True

        if remove_from_car:
            self.car_graph.remove_edge(*edge_to_transform)

        e = edge_to_transform[:2]
        orig_capacity = self.total_capacities[edge_to_transform[:2]]
        remaining_car_capacity = orig_capacity - 1
        car_capacity_straight = remaining_car_capacity // 2
        
        if (car_capacity_straight == 0 and remaining_car_capacity > 0 and 
            self.car_graph.number_of_edges(edge_to_transform[1], edge_to_transform[0]) == 0):
            car_capacity_straight = remaining_car_capacity

        self.fixed_capacities.loc[-1] = {"Edge": (e[1], e[0]), "u_b(e)": 1, "capacity": orig_capacity, "u_c(e)": remaining_car_capacity - car_capacity_straight}
        self.fixed_capacities.loc[-2] = {"Edge": (e[0], e[1]), "u_b(e)": 1, "capacity": orig_capacity, "u_c(e)": car_capacity_straight}
        self.fixed_capacities.index = self.fixed_capacities.index + 2

    def add_to_pareto(self, bike_edges, edges_removed):
        weight_od_flow = self.optimize_kwargs.get("weight_od_flow", False)
        bike_travel_time, car_travel_time = compute_travel_times_in_graph(
            self.modified_G_lane, self.od, self.sp_method, weight_od_flow
        )
        self.pareto_df.append({
            "bike_edges_added": edges_removed,
            "bike_edges": bike_edges,
            "car_edges": self.car_graph.number_of_edges(),
            "bike_time": bike_travel_time,
            "car_time": car_travel_time,
        })
        print(f"Step {edges_removed}: BikeTime={bike_travel_time:.2f}, CarTime={car_travel_time:.2f}")

    def pareto(self, save_graph_path=None, fix_multilane=True, rl_model_path=None):
        """
        Extended Pareto function. 
        If rl_model_path is provided, it uses the RL Agent to rank edges instead of the LP solver.
        """
        self.reset_pareto_variables()
        is_fixed_car = nx.get_edge_attributes(self.G_lane, "fixed")
        self.add_to_pareto(0, 0)

        # 1. Pre-fix multilane edges
        if fix_multilane:
            edges_to_fix = fix_multilane_bike_lanes(self.G_lane, check_for_existing=False)
            for e in edges_to_fix:
                if not is_fixed_car.get(e, False):
                    self.allocate_bike_edge(e, assert_greater_0=True, remove_from_car=True)
            self.add_to_pareto(len(edges_to_fix), 0)
        else:
            edges_to_fix = []

        # 2. Load RL model if path provided
        rl_model = None
        if rl_model_path:
            # Remove .zip if the user manually added it, because PPO.load adds it back
            model_path = rl_model_path
            if model_path.endswith(".zip"):
                model_path = model_path[:-4]
            
            print(f"Using RL Agent from {model_path} for edge ranking...")
            rl_model = PPO.load(model_path, device="cpu")

        edges_removed = 0
        found_edge = True
        cap_sorted = None

        while found_edge:
            # Re-optimize/Re-rank every x steps
            if edges_removed % self.optimize_every_x == 0:
                if rl_model:
                    # RL OPTION: Predict edge rankings using the agent
                    cap_sorted = self._get_rl_edge_rankings(rl_model)
                else:
                    # LP OPTION: Standard LP Solver
                    capacities = self.optimize(self.fixed_capacities)
                    cap_sorted = capacities.sort_values(["u_b(e)", "u_c(e)"], ascending=[False, True])

            found_edge = False
            for e in cap_sorted["Edge"]:
                if (e not in self.modified_G_lane.edges()) or self.is_bike[e]:
                    continue
                
                for key in list(dict(self.modified_G_lane[e[0]][e[1]])):
                    edge_to_transform = (e[0], e[1], key)
                    if not is_fixed_car.get(edge_to_transform, False):
                        self.car_graph.remove_edge(*edge_to_transform)
                        if not nx.is_strongly_connected(self.car_graph):
                            self.car_graph.add_edge(*edge_to_transform)
                            is_fixed_car[edge_to_transform] = True
                            continue
                        else:
                            found_edge = True
                            break
                if found_edge: break
            
            if not found_edge: break

            edges_removed += 1
            self.allocate_bike_edge(edge_to_transform)
            self.add_to_pareto(len(edges_to_fix) + edges_removed, edges_removed)
        # Save intermediate graphs
            if save_graph_path is not None and edges_removed % self.optimize_every_x == 0:
                if edges_removed > 20:
                    graph_file = f"{save_graph_path}_graph_{edges_removed}.csv"
                    edge_df = nx.to_pandas_edgelist(self.modified_G_lane, edge_key="edge_key")[
                        ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
                    ]
                    edge_df.to_csv(graph_file, index=False)
                    print(f"Saved graph: {graph_file}")
                    
        return pd.DataFrame(self.pareto_df)

    def _get_rl_edge_rankings(self, model):
        """
        Returns a DataFrame with edges sorted by RL predicted value.
        Handles dimension mismatches by padding or truncating the observation.
        """
        # Use the original lane edge list (with keys) to match training
        lane_edges = list(self.G_lane.edges(keys=True))
        
        # Build observation vector (same as in BikeNetworkEnv._get_obs)
        obs = []
        for u, v, k in lane_edges:
            data = self.modified_G_lane[u][v][k] if self.modified_G_lane.has_edge(u, v, k) else self.G_lane[u][v][k]
            is_bike = 1.0 if data.get("lanetype") == "P" else 0.0
            obs.extend([float(data.get('distance', 0)), 
                        float(data.get('gradient', 0)), 
                        is_bike])
        
        obs = np.array(obs, dtype=np.float32)
        
        # --- NEW: Handle dimension mismatches (padding/truncation) ---
        expected_dim = model.observation_space.shape[0]
        current_dim = obs.shape[0]
        if current_dim != expected_dim:
            print(f"Warning: observation dim mismatch: {current_dim} vs {expected_dim}. Padding/truncating.")
            if current_dim < expected_dim:
                pad_size = expected_dim - current_dim
                obs = np.pad(obs, (0, pad_size), 'constant')
            else:
                obs = obs[:expected_dim]
        # -----------------------------------------------------------
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(model.device)
        obs_tensor = obs_tensor.view(1, -1)  # shape (1, N*3)
        
        # Forward pass to get action values
        with torch.no_grad():
            features = model.policy.extract_features(obs_tensor)
            latent_pi, _ = model.policy.mlp_extractor(features)
            action_values = model.policy.action_net(latent_pi).squeeze().cpu().numpy()
        
        # Map action values (per lane edge) to street edges (undirected)
        street_edge_values = {}
        for (u, v, k), value in zip(lane_edges, action_values):
            # Use undirected (u, v) as key, keep max value
            key = (u, v) if u < v else (v, u)
            if key not in street_edge_values or value > street_edge_values[key]:
                street_edge_values[key] = value
        
        # Create DataFrame sorted by value descending
        ranking_df = pd.DataFrame([
            {"Edge": key, "u_b(e)": val} for key, val in street_edge_values.items()
        ]).sort_values("u_b(e)", ascending=False)
        
        return ranking_df