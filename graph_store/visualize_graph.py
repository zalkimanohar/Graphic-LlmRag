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
    Builds an F1 Knowledge Graph from the Star Schema dataset 
    (dim_races, fact_session_results, dim_constructors, dim_drivers),
    exports Gephi GEXF graph files, and renders high-res PNG plots.
    """
    def __init__(self, data_dir: str = "data", uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.data_dir = data_dir
        self.uri = uri
        self.user = user
        self.password = password

    def build_graph_from_parquet(self) -> nx.DiGraph:
        """Loads Star Schema parquet files and constructs NetworkX graph topology."""
        G = nx.DiGraph()
        
        # Resolve data directory path
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
            print("--- Building Knowledge Graph from local Gold Parquet datasets ---")
            df_fact = pd.read_parquet(fact_files[0])
            df_drivers = pd.read_parquet(drivers_files[0])
            df_constructors = pd.read_parquet(constructors_files[0])
            df_races = pd.read_parquet(races_files[0])

            # 1. Add Driver Nodes
            for _, row in df_drivers.iterrows():
                d_id = f"Driver:{row['driver_id']}"
                G.add_node(d_id, label="Driver", name=str(row.get('driver_name', row['driver_id'])), nationality=str(row.get('nationality', '')))

            # 2. Add Constructor Nodes
            for _, row in df_constructors.iterrows():
                c_id = f"Constructor:{row['constructor_id']}"
                G.add_node(c_id, label="Constructor", name=str(row.get('constructor_name', row['constructor_id'])), nationality=str(row.get('nationality', '')))

            # 3. Add Race Nodes
            for _, row in df_races.iterrows():
                r_id = f"Race:{row['season']}_{row['round']}"
                G.add_node(r_id, label="Race", name=str(row.get('race_name', f"Round {row['round']}")), season=int(row['season']))

            # 4. Add Fact Relationships
            for _, row in df_fact.head(100).iterrows():
                d_id = f"Driver:{row['driver_id']}"
                c_id = f"Constructor:{row['constructor_id']}"
                r_id = f"Race:{row['season']}_{row['round']}"

                # Relationships based on Star Schema FKs
                G.add_edge(d_id, c_id, label="DRIVES_FOR")
                G.add_edge(c_id, r_id, label="COMPETED_IN")

                if row.get('is_win') == 1:
                    G.add_edge(d_id, r_id, label="WON_RACE")
                elif row.get('is_podium') == 1:
                    G.add_edge(d_id, r_id, label="PODIUM_FINISH")
                else:
                    G.add_edge(d_id, r_id, label="PARTICIPATED_IN")

            print(f"✅ Built Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} relationships.")
            return G
        else:
            print("⚠️ Local Parquet data not found. Fetching from Neo4j or generating Schema fallback graph...")
            return self.fetch_graph_from_neo4j_or_fallback()

    def fetch_graph_from_neo4j_or_fallback(self) -> nx.DiGraph:
        """Connects to Neo4j database or builds fallback schema representation."""
        G = nx.DiGraph()
        if NEO4J_AVAILABLE:
            try:
                driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with driver.session() as session:
                    res = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
                    for rec in res:
                        n, m, r = rec["n"], rec["m"], rec["r"]
                        n_id = f"{list(n.labels)[0]}:{n.get('driver_id', n.get('constructor_id', n.get('name', 'N')))}"
                        m_id = f"{list(m.labels)[0]}:{m.get('driver_id', m.get('constructor_id', m.get('name', 'M')))}"
                        rel_type = type(r).__name__ if hasattr(r, '__name__') else "CONNECTED"
                        G.add_edge(n_id, m_id, label=rel_type)
                driver.close()
                if G.number_of_nodes() > 0:
                    return G
            except Exception as e:
                print(f"⚠️ Neo4j Offline ({e}). Falling back to visual schema network.")

        # Fallback F1 Star Schema structural mock graph
        schema_edges = [
            ("Driver:Max_Verstappen", "Constructor:Red_Racing", "DRIVES_FOR"),
            ("Driver:Charles_Leclerc", "Constructor:Ferrari", "DRIVES_FOR"),
            ("Driver:Lewis_Hamilton", "Constructor:Ferrari", "DRIVES_FOR"),
            ("Constructor:Red_Racing", "Race:Monaco_GP", "COMPETED_IN"),
            ("Constructor:Ferrari", "Race:Monaco_GP", "COMPETED_IN"),
            ("Driver:Max_Verstappen", "Race:Monaco_GP", "WON_RACE"),
            ("Driver:Charles_Leclerc", "Race:Monaco_GP", "PODIUM_FINISH"),
            ("Driver:Lewis_Hamilton", "Race:Monaco_GP", "PARTICIPATED_IN")
        ]
        for src, tgt, rel in schema_edges:
            G.add_edge(src, tgt, label=rel)
        return G

    def export_gephi_files(self, G: nx.DiGraph):
        """Exports GEXF and GraphML files for Gephi and Neo4j Desktop imports."""
        gexf_path = "f1_graph.gexf"
        graphml_path = "f1_graph.graphml"
        
        nx.write_gexf(G, gexf_path)
        print(f"✅ Gephi GEXF export file saved: `{gexf_path}`")
        
        nx.write_graphml(G, graphml_path)
        print(f"✅ Standard GraphML file saved: `{graphml_path}`")

    def plot_and_save_png(self, G: nx.DiGraph):
        """Renders network visualization and saves publication PNG."""
        if G.number_of_nodes() == 0:
            print("⚠️ Cannot plot empty graph.")
            return

        fig = plt.figure(figsize=(14, 9))
        plt.title("🏁 F1 Analytics Knowledge Graph (Star Schema Mapping)", fontsize=16, color="white", pad=25)

        pos = nx.spring_layout(G, k=0.35, seed=42)

        # Categorize node colors
        node_colors = []
        for n in G.nodes():
            if "Driver" in n:
                node_colors.append("#e10600")       # F1 Red
            elif "Constructor" in n:
                node_colors.append("#00d2be")   # Teal
            elif "Race" in n:
                node_colors.append("#ffb800")          # Gold/Yellow
            else:
                node_colors.append("#888888")

        nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.6, alpha=0.6, edge_color="#cccccc", arrows=True, arrowsize=16)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")

        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#00ffcc")

        ax = plt.gca()
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")
        plt.box(False)

        png_path = "f1_knowledge_graph.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✅ Graph PNG rendering saved: `{png_path}`")
        plt.show()


if __name__ == "__main__":
    visualizer = F1StarSchemaGraphVisualizer()
    graph = visualizer.build_graph_from_parquet()
    visualizer.export_gephi_files(graph)
    visualizer.plot_and_save_png(graph)