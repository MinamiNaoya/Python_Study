import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()
G.add_edges_from([(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1), (5, 10), (6, 3)])

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_labels(G, pos)

plt.title("Directed Acyclic Graph")
plt.axis("off")
plt.show()

# トポロジカルソートする
topological_order = list(nx.topological_sort(G))
print("トポロジカルソート結果：", topological_order)

H = G.subgraph(topological_order)

pos = nx.spring_layout(H)
nx.draw_networkx_nodes(H, pos, node_size=700)
nx.draw_networkx_edges(H, pos, arrows=True)
nx.draw_networkx_labels(H, pos)
plt.title("Topologically Sorted Directed Acyclic Graph")
plt.axis("off")
plt.show()