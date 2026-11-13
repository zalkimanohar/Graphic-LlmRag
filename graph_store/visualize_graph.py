import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class F1StarSchemaGraphVisualizer:
    """
    Builds an F1 Knowledge Graph from Star Schema datasets,
    exports Gephi GEXF graph files, and renders publication-ready PNG plots.
    """
    def __init__(self, data_dir: str = "data", uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.data_dir = data_dir
        self.uri = uri
        self.user = user
        self.password = password

    def build_graph_from_parquet(self) -> nx.DiGraph:
        G = nx.DiGraph()
        target_dir = self.data_dir
        if not os.path.exists(target_dir):
            if os.path.exists("../data"):
                target_dir = "../data"
            elif os.path.exists("data"):
                target_dir = "data"

        fact_files = glob.glob(os.path.join(target_dir, "fact_session_results", "*.parquet"))
        drivers_files = glob.glob(os.path.join(target_dir, "dim_drivers", "*.parquet"))
        constructors_files = glob.glob(os.path.join(target_dir, "dim_constructors", "*.parquet"))
        races_files = glob.glob(os.path.join(target_dir, "dim_races", "*.parquet"))

        if fact_files and drivers_files and constructors_files and races_files:
            df_fact = pd.read_parquet(fact_files[0])
            df_drivers = pd.read_parquet(drivers_files[0])
            df_constructors = pd.read_parquet(constructors_files[0])
            df_races = pd.read_parquet(races_files[0])

            for _, row in df_drivers.iterrows():
                d_id = f"Driver:{row['driver_id']}"
                G.add_node(d_id, label="Driver", name=str(row.get('driver_name', row['driver_id'])))

            for _, row in df_constructors.iterrows():
                c_id = f"Constructor:{row['constructor_id']}"
                G.add_node(c_id, label="Constructor", name=str(row.get('constructor_name', row['constructor_id'])))

            for _, row in df_races.iterrows():
                r_id = f"Race:{row['season']}_{row['round']}"
                G.add_node(r_id, label="Race", name=str(row.get('race_name', f"Round {row['round']}")))

            for _, row in df_fact.head(100).iterrows():
                d_id = f"Driver:{row['driver_id']}"
                c_id = f"Constructor:{row['constructor_id']}"
                r_id = f"Race:{row['season']}_{row['round']}"

                G.add_edge(d_id, c_id, label="DRIVES_FOR")
                G.add_edge(c_id, r_id, label="COMPETED_IN")
                if row.get('is_win') == 1:
                    G.add_edge(d_id, r_id, label="WON_RACE")
                elif row.get('is_podium') == 1:
                    G.add_edge(d_id, r_id, label="PODIUM_FINISH")
            return G
        return self.fetch_fallback_graph()

    def fetch_fallback_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        schema_edges = [
            ("Driver:Max_Verstappen", "Constructor:Red_Bull", "DRIVES_FOR"),
            ("Driver:Charles_Leclerc", "Constructor:Ferrari", "DRIVES_FOR"),
            ("Constructor:Red_Bull", "Race:Monaco_GP", "COMPETED_IN"),
            ("Constructor:Ferrari", "Race:Monaco_GP", "COMPETED_IN"),
            ("Driver:Max_Verstappen", "Race:Monaco_GP", "WON_RACE")
        ]
        for src, tgt, rel in schema_edges:
            G.add_edge(src, tgt, label=rel)
        return G

    def export_gephi_files(self, G: nx.DiGraph):
        nx.write_gexf(G, "f1_graph.gexf")
        nx.write_graphml(G, "f1_graph.graphml")

    def plot_and_save_png(self, G: nx.DiGraph):
        if G.number_of_nodes() == 0:
            return
        fig = plt.figure(figsize=(14, 9))
        plt.title("🏁 F1 Analytics Knowledge Graph", fontsize=16, color="white", pad=25)
        pos = nx.spring_layout(G, k=0.35, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=1100, node_color="#e10600", alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.6, alpha=0.6, edge_color="#cccccc", arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")
        ax = plt.gca()
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")
        plt.savefig("f1_knowledge_graph.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

if __name__ == "__main__":
    vis = F1StarSchemaGraphVisualizer()
    g = vis.build_graph_from_parquet()
    vis.export_gephi_files(g)
    vis.plot_and_save_png(g)