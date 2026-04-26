import time
import os
import json
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import sys

# Maintain your SNMan pathing
sys.path.append('/paper/bike_lane_optimization_rl/snman') 
import snman

from ebike_city_tools.graph_utils import (
    lane_to_street_graph,
    keep_only_the_largest_connected_component,
    load_lane_graph,
)
from ebike_city_tools.optimize.wrapper import generate_motorized_lane_graph
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.optimize.rounding_utils import combine_paretos_from_path, combine_pareto_frontiers
from ebike_city_tools.iterative_algorithms import betweenness_pareto, topdown_betweenness_pareto
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

# --- CONSTANTS ---
ROUNDING_METHOD = "round_bike_optimize"
IGNORE_FIXED = True
FIX_MULTILANE = True
FLOW_CONSTANT = 1 
WEIGHT_OD_FLOW = False
RATIO_BIKE_EDGES = 0.4

algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="../street_network_data", type=str)
    parser.add_argument("-i", "--instance", default="affoltern", type=str)
    parser.add_argument("-o", "--out_path", default="outputs", type=str)
    parser.add_argument("-k", "--optimize_every_k", default=50, type=int, help="how often to re-optimize")
    parser.add_argument("-c", "--car_weight", default=1, type=float, help="weighting of cars in objective function")
    parser.add_argument("-v", "--valid_edges_k", default=0, type=int, help="subsampling nodes around SP")
    parser.add_argument("-p", "--penalty_shared", default=2, type=int, help="shared lane factor")
    parser.add_argument("-s", "--sp_method", default="od", type=str, help="all_pairs or od")
    parser.add_argument("--save_graph", action="store_true", help="save graph after optimization")
    parser.add_argument("-a", "--algorithm", type=str, default="optimize", 
                        help="optimize, betweenness_topdown, betweenness_cartime, betweenness_biketime")
    
    # --- ADDED RL ARGUMENT ---
    parser.add_argument("-m", "--model_path", default=None, type=str, help="Path to trained RL model (.zip)")
    
    args = parser.parse_args()

    # --- Setup Paths ---
    instance_name = args.instance
    path = os.path.join(args.data_path, instance_name)
    shared_lane_factor = args.penalty_shared
    out_path = os.path.join(args.out_path, args.instance)
    os.makedirs(out_path, exist_ok=True)
    
    np.random.seed(42)

    # --- Load Graph ---
    G_lane = load_lane_graph(path)
    G_lane = keep_only_the_largest_connected_component(G_lane)
    
    # --- Load OD ---
    od = pd.read_csv(os.path.join(path, "od_matrix.csv"))
    od = od[od["s"] != od["t"]]
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]

    assert nx.is_strongly_connected(G_lane), "G not connected"

    # --- Handle Betweenness Algorithms ---
    if "betweenness" in args.algorithm:
        print(f"Running betweenness algorithm {args.algorithm}")
        algorithm_func, kwargs = algorithm_dict[args.algorithm]
        save_path = os.path.join(out_path, f"{args.algorithm}") if args.save_graph else None

        pareto_between = algorithm_func(
            G_lane.copy(),
            sp_method=args.sp_method,
            od_matrix=od,
            weight_od_flow=WEIGHT_OD_FLOW,
            fix_multilane=FIX_MULTILANE,
            save_graph_path=save_path,
            save_graph_every_x=args.optimize_every_k,
            **kwargs,
        )
        pareto_between.to_csv(os.path.join(out_path, f"real_pareto_{args.algorithm}.csv"), index=False)
        sys.exit()

    # --- Handle Optimization (LP or RL) ---
    assert args.algorithm == "optimize"
    od = extend_od_circular(od, node_list)

    runtimes_pareto = []
    car_weight = float(args.car_weight)
    
    # Use same naming convention as LP version (based on sp_method)
    out_path_ending = "_od" if args.sp_method == "od" else ""
    fn_parameters = f"optimize{out_path_ending}_{car_weight}_{args.optimize_every_k}"
    mode_label = "rl" if args.model_path else "lp"

    print(f"Starting Pareto Optimization ({mode_label.upper()} Mode)...")
    tic = time.time()

    # Initialize Pareto optimizer
    opt = ParetoRoundOptimize(
        G_lane.copy(),
        od.copy(),
        optimize_every_x=args.optimize_every_k,
        car_weight=car_weight,
        sp_method=args.sp_method,
        shared_lane_factor=shared_lane_factor,
        weight_od_flow=WEIGHT_OD_FLOW,
        valid_edges_k=args.valid_edges_k,
    )

    # Run pareto (now supporting the RL path)
    save_graph_path = os.path.join(out_path, fn_parameters) if args.save_graph else None
    print(f"save_graph_path = {save_graph_path}")   # <-- add this line
    pareto_df = opt.pareto(
        fix_multilane=FIX_MULTILANE, 
        save_graph_path=save_graph_path,
        rl_model_path=args.model_path  # The critical link to your RL Agent
    )

    # --- Save Results & Runtimes ---
    print(f"Finished. Pareto frontier found with {len(pareto_df)} solutions.")
    total_time = time.time() - tic
    print(f"Total runtime: {total_time:.2f} seconds")
    
    runtime_dict = opt.runtimes
    runtime_dict["time_pareto"] = total_time
    runtimes_pareto.append(runtime_dict)
   
    pareto_df.to_csv(os.path.join(out_path, f"real_pareto_{fn_parameters}.csv"), index=False)

    with open(os.path.join(out_path, f"runtime_pareto_{fn_parameters}.json"), "w") as outfile:
        json.dump(runtimes_pareto, outfile)

    # Combine all results in the folder
    combined_pareto = combine_pareto_frontiers(combine_paretos_from_path(out_path))
    combined_pareto.to_csv(os.path.join(out_path, f"real_pareto_combined_optimize.csv"), index=False)