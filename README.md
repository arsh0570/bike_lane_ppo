# Transferable Bike Network Planning with Reinforcement Learning

This repository for bike lane reallocation in urban street networks using PPO RL agent that learns a transferable policy. After training on one district (e.g., Affoltern), the same agent can generate bike lane networks for other districts (e.g., Birchplatz) with negligible inference cost.

The agent sequentially converts car lanes into bike lanes while maintaining car network connectivity. The reward function balances bike travel time improvement against car travel time penalties. We provide:
- **PPO** (Proximal Policy Optimization) using a flat state vector (padding for variable graph sizes). The resulting Pareto frontiers show the trade‑off between bike and car travel times. 
---

## Installation

Clone the repository and install the package in editable mode inside a virtual environment:

```bash
git clone https://github.com/arsh0570/bike_lane_ppo.git
cd bike_lane_rl
python -m venv env
source env/bin/activate          # Linux/Mac
# or .\env\Scripts\activate on Windows
pip install -e .
```

<img src="assets/overview.png" alt="Pareto frontier and corresponding street networks" style="width:100%;">


This installs the package called `ebike_city_tools` by [1] in the virtual environment, together with all dependencies, including stable-baselines3 for PPO.


Most scripts will save the results to the `outputs` folder by default.

### Running the algorithm on real data

Two instances of street networks (Affoltern, Birchplatz) in the city of Zurich can be downloaded [here](https://polybox.ethz.ch/index.php/s/YaoJkHRofZKUiTG).

To preprocess the data, the [SNMan](https://github.com/lukasballo/snman) package is required. Installation instructions can be found in their repo.

After downloading the data and installing SNMan, PPO algorithm can be trained by Navigating to the scripts/ folder and running the training script:

```
cd scripts
python train_rl_agent.py --instance affoltern --total_timesteps 10000 --output_dir ../models
```
more optional argumnets can be found in the file train-rl_agent,py

after training the PPO agent the algorithm can be executed by running `run_real_data_rl.py` with the path to the data specified via the `-d` flag.

Usage:
```
python scripts/run_real_data.py [-h] [-d DATA_PATH] [-i INSTANCE] [-o OUT_PATH] [-k OPTIMIZE_EVERY_K] [-c CAR_WEIGHT] [-v VALID_EDGES_K] [-p PENALTY_SHARED] [-s SP_METHOD] [--save_graph] [-a ALGORITHM] [-m MODEL PATH]

optional arguments:
 -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
  -i INSTANCE, --instance INSTANCE
  -o OUT_PATH, --out_path OUT_PATH
  -k OPTIMIZE_EVERY_K, --optimize_every_k OPTIMIZE_EVERY_K
                        how often to re-optimize
  -c CAR_WEIGHT, --car_weight CAR_WEIGHT
                        weighting of cars in objective function
  -v VALID_EDGES_K, --valid_edges_k VALID_EDGES_K
                        if subsampling edges, the number of nodes around SP
  -p PENALTY_SHARED, --penalty_shared PENALTY_SHARED
                        penalty factor for driving on a car lane by bike
  -s SP_METHOD, --sp_method SP_METHOD
                        Compute the shortest path either 'all_pairs' or 'od'
  --save_graph          if true, only creating one graph and saving it
  -a ALGORITHM, --algorithm ALGORITHM
                        One of optimize, betweenness_topdown, betweenness_cartime, betweenness_biketime
  -m MODEL PATH,  --model_path MODEL PATH 
                        Path to trained RL model (.zip)
```

#### Visualisation
if the graphs are saved using the --save_graph argument, they can be visiualised as follows-:

Usage:
```
python scripts/visualise_network.py [-h] [-i INSTANCE] [-c CSV] [-d DATA_PATH] [-o OUT_DIR]
                                    [--input_crs INPUT_CRS] [--output_crs OUTPUT_CRS]
                                    [--bike_colour BIKE_COLOUR] [--mixed_colour MIXED_COLOUR]
                                    [--car_colour CAR_COLOUR] [--dpi DPI] [--no_show]

optional arguments:
  -h, --help            show this help message and exit
  -i INSTANCE, --instance INSTANCE
                        Instance name (subfolder in data and outputs)
  -c CSV, --csv CSV     Path to the optimised edges CSV file. If just a filename,
                        assumes outputs/INSTANCE/
  -d DATA_PATH, --data_path DATA_PATH
                        Root data directory (contains instance subfolder)
  -o OUT_DIR, --out_dir OUT_DIR
                        Directory to save plots. If None, uses current script directory
  --input_crs INPUT_CRS
                        CRS of the original graph coordinates (e.g. EPSG:2056 for Swiss LV95)
  --output_crs OUTPUT_CRS
                        CRS for plotting (Web Mercator)
  --bike_colour BIKE_COLOUR
                        Colour for bike lanes (default: #FF2525)
  --mixed_colour MIXED_COLOUR
                        Colour for mixed traffic lanes (default: gray)
  --car_colour CAR_COLOUR
                        Colour for car lanes (default: #3badff)
  --dpi DPI             DPI for saved figures (default: 300)
  --no_show             Do not display plots (only save them)
```

References

[1] Wiedemann, N., Nöbel, C., Martin, H., Ballo, L., & Raubal, M. (2024). Bike network planning in limited urban space. arXiv preprint arXiv:2405.01770.
