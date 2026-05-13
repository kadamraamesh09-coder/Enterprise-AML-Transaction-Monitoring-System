# src/network_graph.py

import pandas as pd
import networkx as nx
from collections import deque


def build_transaction_graph(tx: pd.DataFrame):
    """
    Build a fast transaction graph using NetworkX DiGraph.
    Optimized to avoid slow iterrows().
    """
    G = nx.DiGraph()

    # Collect edges as tuples instead of iterrows
    edges = list(zip(tx["customer_id"], tx["counterparty_id"], tx["amount"]))

    # Add edges in a single batch (much faster)
    for src, dst, amt in edges:
        G.add_edge(src, dst, amount=amt)

    return G


def detect_multi_hop_layering(G, max_hops=4):
    """
    Detect multi-hop layering using BFS traversal.
    Only finds actual transaction paths (A→B→C→D).
    Avoids checking every pair of nodes.

    Returns pandas DataFrame with layering paths.
    """

    layering_paths = []

    # BFS from each node but with limited depth (max_hops)
    for start in G.nodes():

        queue = deque([(start, [start])])

        while queue:
            node, path = queue.popleft()

            if len(path) - 1 >= 2:  # at least 2-hop chain
                layering_paths.append({
                    "start": start,
                    "end": node,
                    "hops": len(path) - 1,
                    "path": " → ".join(path)
                })

            if len(path) - 1 == max_hops:
                continue  # stop deeper search

            for neighbor in G.neighbors(node):
                if neighbor not in path:  # avoid cycles
                    queue.append((neighbor, path + [neighbor]))

    # Convert to DataFrame
    df = pd.DataFrame(layering_paths)

    # Remove duplicates
    if not df.empty:
        df = df.drop_duplicates(subset=["path"])

    return df
