import networkx as nx


def color_communities(G: nx.Graph):
    # Use the FastGreedy algorithm to identify communities in the network
    communities = list(
        nx.algorithms.community.greedy_modularity_communities(G))

    # Print the communities
    print(f'Number of communities: {len(communities)}')
    for i, community in enumerate(communities):
        print(f'Community {i+1}: {community}')

    # Create a mapping from nodes to community IDs
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    colors = [node_to_community[n] for n in G.nodes()]
    return colors
