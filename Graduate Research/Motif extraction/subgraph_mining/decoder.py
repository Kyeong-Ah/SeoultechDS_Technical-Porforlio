import argparse
import csv
from itertools import combinations
import time
import os

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
# ★ 재현성 보장 위해 난수 고정. For REPRODUCIBILITY
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ★ 모티프에 노드 순서 반영하기 위한 함수
def assign_order_attrs(G, anchor=0):
    H = G.copy()
    if H.has_edge(anchor, anchor):  # self-loop는 정렬에서 제외
        H.remove_edge(anchor, anchor)

    # 1) DAG이면 토폴로지 순서
    if H.is_directed() and nx.is_directed_acyclic_graph(H):
        order = list(nx.topological_sort(H))
        rank = {v: i for i, v in enumerate(order)}
    else:
        # 2) 일반 그래프: 앵커 기준 (무향) BFS 거리 → tie-break
        UG = H.to_undirected() if H.is_directed() else H
        dist = nx.single_source_shortest_path_length(UG, anchor)
        # 정의되지 않은 노드는 큰 값으로
        big = max(dist.values()) + 1 if dist else 0
        def key(v):
            d = dist.get(v, big)
            return (d, -H.in_degree(v) if H.is_directed() else 0, v)
        order = sorted(H.nodes(), key=key)
        rank = {v: i for i, v in enumerate(order)}

    nx.set_node_attributes(G, rank, name="rank")
    return G, rank


def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    # PATTERN 1
    pattern = generator.generate(size=10)
    # PATTERN 2
    #pattern = nx.star_graph(9)
    # PATTERN 3
    #pattern = nx.complete_graph(10)
    # PATTERN 4
    #pattern = nx.Graph()
    #pattern.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    #    (6, 7), (7, 2), (7, 8), (8, 9), (9, 10), (10, 6)])
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs

# === ADD: NetworkX 보존 + PyG 객체만 변환하는 안전 래퍼 ===
def _ensure_nx(g):
    # 이미 NetworkX면 그대로 반환
    if isinstance(g, (nx.Graph, nx.DiGraph)):
        return g
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        # 병렬 엣지가 있으면 단순화 (필요 시 유지해도 무방)
        return nx.DiGraph(g) if g.is_directed() else nx.Graph(g)
    # PyG의 Data/HeteroData 등만 변환
    return pyg_utils.to_networkx(g, to_undirected=False)  # 9/20 : to_undirected=False 만 추가


def pattern_growth(dataset, task, args):
    start_time = time.time()
    # init model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    # load data
    neighs_pyg, neighs = [], []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": print("using label 0")
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0:
            continue
        if task == "graph-truncate" and i >= 1000: 
            break
        # if not type(graph) == nx.Graph:
        #     # graph = pyg_utils.to_networkx(graph).to_undirected() # 기존 무방향성 코드
        #     graph = pyg_utils.to_networkx(graph) # 방향성 유지 # 이 if 문 안 쓰고 아래 _ensure_nx() 로 대체한 것
        graph = _ensure_nx(graph)
        graphs.append(graph)
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        # 반경은 무향 도달성으로 계산(방향으로 막히는 것 방지)
                        # neigh = list(nx.single_source_shortest_path_length(graph, node, cutoff=args.radius).keys()) # 기존 코드
                        # ↓↓ 아래 두 줄이 수정된 코드
                        src_graph = graph.to_undirected() if graph.is_directed() else graph
                        neigh = list(nx.single_source_shortest_path_length(src_graph, node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            # neigh = neigh.subgraph(max(nx.connected_components(neigh), key=len)) # 기존 코드.
                            # ↓↓ 아래 if 문이 수정된 코드.
                            if neigh.is_directed(): #연결 성분: DiGraph면 weakly, 무향이면 connected 사용
                                cc = max(nx.weakly_connected_components(neigh), key=len)
                            else:
                                cc = max(nx.connected_components(neigh), key=len)
                            neigh = neigh.subgraph(cc).copy()
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        # ★ 순서 속성 부여
                        neigh, _ = assign_order_attrs(neigh, anchor=0)
                        neighs.append(neigh)
        elif args.sample_method == "tree":
            start_time = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)   # after converting labels, 0 will be anchor

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        #top = min(len(neighs), (i+1)*args.batch_size)
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))

        embs.append(emb)

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size)
    out_graphs = agent.run_search(args.n_trials)
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    # visualize out patterns
    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        # if args.node_anchored: # 기존 코드. 아래 if 문 두 개로 대체됨.
        #     colors = ["red"] + ["blue"]*(len(pattern)-1)
        #     nx.draw(pattern, node_color=colors, with_labels=True)
        # else:
        #     nx.draw(pattern)

        '''# ★ rank 속성이 없으면(안 붙였으면) 여기서 한 번 부여  
        first_attrs = next(iter(pattern.nodes(data=True)))[1]
        if 'rank' not in first_attrs:
            pattern, _ = assign_order_attrs(pattern, anchor=0 if args.node_anchored else list(pattern.nodes())[0])

        if args.node_anchored:
            labels = {n: ("0★" if n == 0 else pattern.nodes[n].get("rank", n)) for n in pattern.nodes()}
            colors = ["red" if n == 0 else "blue" for n in pattern.nodes()]
        else:
            labels = {n: pattern.nodes[n].get("rank", n) for n in pattern.nodes()}
            colors = "blue"'''   # 수정된 코드.

        # ▶▶ 9/20 수정한 코드
        # 1) 라벨: 실제 노드 이름
        labels = {n: str(n) for n in pattern.nodes()}
        # 2) 색: 앵커 속성(anchor=1)은 빨강, 나머지 파랑
        colors = ["red" if pattern.nodes[n].get("anchor", 0) == 1 else "blue"
                for n in pattern.nodes()]
        
        nx.draw(pattern, with_labels=True, labels=labels, node_color=colors, arrows=pattern.is_directed())

        print("Saving plots/cluster/{}-{}.png".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.png".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.pdf".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.close()
        count_by_size[len(pattern)] += 1

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()
#    args.dataset = "enzymes" # 사용자 설정 데이터셋 사용하도록 수정

    print("Using dataset {}".format(args.dataset))
    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset.startswith('zap-graphml:'):
        gpath = args.dataset.split(':', 1)[1]
        G = nx.read_graphml(gpath)
        # # SPMiner 원본 파이프라인은 무향 기반으로 돌아갑니다.
        # # 방향 보존하려면 별도 수정 필요(아래 참고). 일단 무향으로:
        # if G.is_directed():
        #     G = G.to_undirected()
        dataset = [G]
        task = 'graph'
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
        task = 'graph'
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        task = 'graph'
    elif args.dataset == 'dblp':
        dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
        task = 'graph-truncate'
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
        task = 'graph'
    elif args.dataset.startswith('roadnet-'):
        graph = nx.Graph()
        with open("data/{}.txt".format(args.dataset), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'

    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()

