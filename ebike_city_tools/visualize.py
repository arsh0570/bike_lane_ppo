import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from ebike_city_tools.graph_utils import load_lane_graph, keep_only_the_largest_connected_component

def main():
    parser = argparse.ArgumentParser(description="Visualise bike and car networks from optimisation results.")
    parser.add_argument("-i", "--instance", default="affoltern", help="Instance name (subfolder in data and outputs)")
    parser.add_argument("-c", "--csv", help="Path to the optimised edges CSV file. If just a filename, assumes outputs/INSTANCE/")
    parser.add_argument("-d", "--data_path", default="data", help="Root data directory (contains instance subfolder)")
    parser.add_argument("-o", "--out_dir", default=None, help="Directory to save plots. If None, uses current script directory.")
    parser.add_argument("--input_crs", default="EPSG:2056", help="CRS of the original graph coordinates (e.g. EPSG:2056 for Swiss LV95)")
    parser.add_argument("--output_crs", default="EPSG:3857", help="CRS for plotting (Web Mercator)")
    parser.add_argument("--bike_colour", default="#FF2525", help="Colour for bike lanes")
    parser.add_argument("--mixed_colour", default="gray", help="Colour for mixed traffic lanes")
    parser.add_argument("--car_colour", default="#3badff", help="Colour for car lanes")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--no_show", action="store_true", help="Do not display plots (only save them)")
    args = parser.parse_args()

    INSTANCE = args.instance
    DATA_PATH = os.path.join(args.data_path, INSTANCE)

    # --- CSV file handling: if bare filename, prepend outputs/INSTANCE/ ---
    if args.csv:
        CSV_FILE = args.csv
        if os.path.sep not in CSV_FILE:          # no directory separator -> bare filename
            CSV_FILE = os.path.join("outputs", "ppo", "bike_lane_agent_affoltern_20260415_0240", INSTANCE, CSV_FILE)
    else:
        CSV_FILE = os.path.join("outputs", "ppo", "bike_lane_agent_affoltern_20260415_0240", INSTANCE, "optimize_od_1.0_50_graph_50.csv")
    CSV_FILE = os.path.abspath(CSV_FILE)
    print(f"Loading optimised edges from {CSV_FILE}")

    # --- Output directory for plots ---
    if args.out_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        OUT_DIR = SCRIPT_DIR
    else:
        OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    INPUT_CRS = args.input_crs
    OUTPUT_CRS = args.output_crs

    # ========== 1. Load original graph and extract coordinates ==========
    print("Loading original lane graph...")
    G_orig = load_lane_graph(DATA_PATH)
    G_orig = keep_only_the_largest_connected_component(G_orig)
    print(f"Original graph has {G_orig.number_of_nodes()} nodes.")

    pos = {}   # original projected coordinates (e.g. LV95)
    for node, data in G_orig.nodes(data=True):
        x = data.get('x') or data.get('lon') or data.get('X')
        y = data.get('y') or data.get('lat') or data.get('Y')
        if x is None and y is None:
            geom = data.get('geometry')
            if geom is not None and hasattr(geom, 'x') and hasattr(geom, 'y'):
                x, y = geom.x, geom.y
        if x is not None and y is not None:
            try:
                pos[node] = (float(x), float(y))
            except (ValueError, TypeError):
                pass
    print(f"Extracted coordinates for {len(pos)} nodes from original graph.")

    # ========== 2. Load optimised edges and build graph ==========
    df = pd.read_csv(CSV_FILE)
    G_opt = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G_opt.add_edge(row['source'], row['target'],
                       key=row['edge_key'],
                       lanetype=row['lanetype'],
                       distance=row['distance'],
                       gradient=row['gradient'])
    print(f"Optimised graph has {G_opt.number_of_nodes()} nodes and {G_opt.number_of_edges()} edges")

    # ========== 3. Map nodes to coordinates ==========
    node_coords = {}
    for node in G_opt.nodes():
        if node in pos:
            node_coords[node] = pos[node]
        else:
            print(f"Warning: Node {node} from CSV not found in original graph.")

    if not node_coords:
        print("No matching nodes found. Exiting.")
        return

    print(f"Found coordinates for {len(node_coords)} out of {G_opt.number_of_nodes()} optimised nodes.")

    # ========== 4. Reproject to Web Mercator ==========
    nodes_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for (x, y) in node_coords.values()],
        crs=INPUT_CRS,
        index=node_coords.keys()
    )
    nodes_gdf_mercator = nodes_gdf.to_crs(OUTPUT_CRS)
    pos_mercator = {node: (geom.x, geom.y) for node, geom in nodes_gdf_mercator.geometry.items()}

    # ========== 5. Keep only nodes with valid (finite) coordinates ==========
    valid_nodes = set()
    for node, (x, y) in pos_mercator.items():
        if np.isfinite(x) and np.isfinite(y):
            valid_nodes.add(node)
        else:
            print(f"Node {node} has invalid coordinates ({x}, {y}) – skipping.")
    print(f"Keeping {len(valid_nodes)} nodes with valid coordinates.")

    # ========== 6. Filter edges by lanetype and valid nodes ==========
    edges_p = []   # bike lanes (P)
    edges_m = []   # mixed traffic lanes (M>)
    for u, v, k, d in G_opt.edges(keys=True, data=True):
        if u not in valid_nodes or v not in valid_nodes:
            continue
        if d['lanetype'] == 'P':
            edges_p.append((u, v))
        elif d['lanetype'] == 'M>':
            edges_m.append((u, v))
    print(f"Edges with lanetype='P': {len(edges_p)}")
    print(f"Edges with lanetype='M>': {len(edges_m)}")

    if not edges_p and not edges_m:
        print("No valid edges to draw. Exiting.")
        return

    # ========== 7. Plot bike network ==========
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    if edges_p:
        nx.draw_networkx_edges(G_opt, pos_mercator, edgelist=edges_p,
                               edge_color=args.bike_colour, width=2.5, alpha=0.9,
                               label='Bike lane (bidirectional)', ax=ax1, arrows=False)
    if edges_m:
        nx.draw_networkx_edges(G_opt, pos_mercator, edgelist=edges_m,
                               edge_color=args.mixed_colour, width=1.5, alpha=0.7,
                               label='Mixed traffic lane', ax=ax1, arrows=False)
    ax1.set_title(f"Bike Network – {INSTANCE.capitalize()}")
    ax1.axis('off')
    ctx.add_basemap(ax1, crs=OUTPUT_CRS, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
    plt.tight_layout()
    bike_out = os.path.join(OUT_DIR, f"{INSTANCE}_bike_network.png")
    plt.savefig(bike_out, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved bike network plot to {bike_out}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig1)

    # ========== 8. Plot car network (all undirected edges) ==========
    car_edges = [(u, v) for u, v in G_opt.edges(keys=False) if u in valid_nodes and v in valid_nodes]
    if car_edges:
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        nx.draw_networkx_edges(G_opt, pos_mercator, edgelist=car_edges,
                               edge_color=args.car_colour, width=2, alpha=0.8,
                               label='Car lanes', ax=ax2, arrows=False)
        ax2.set_title(f"Car Network – {INSTANCE.capitalize()}")
        ax2.axis('off')
        ctx.add_basemap(ax2, crs=OUTPUT_CRS, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        plt.tight_layout()
        car_out = os.path.join(OUT_DIR, f"{INSTANCE}_car_network.png")
        plt.savefig(car_out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved car network plot to {car_out}")
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig2)

if __name__ == "__main__":
    main()