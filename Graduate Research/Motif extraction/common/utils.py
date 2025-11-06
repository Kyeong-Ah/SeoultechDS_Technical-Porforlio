from collections import defaultdict, Counter

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import torch
import torch.optim as optim
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm

from common import feature_preprocess

def is_linear_path(g: 'nx.Graph | nx.DiGraph') -> bool: # 9/20 추가
    """서브그래프 g가 (방향 고려 시) 분기 없는 선형 체인인지 판정."""
    if g is None or len(g) == 0:
        return False
    H = g.copy()
    # self-loop 제거
    H.remove_edges_from(nx.selfloop_edges(H))

    # 연결성 확인(유향이면 약연결성)
    if H.is_directed():
        if not nx.is_weakly_connected(H):
            return False
        n, m = H.number_of_nodes(), H.number_of_edges()
        if n == 1:
            return True
        if m != n - 1:
            return False
        indeg = dict(H.in_degree())
        outdeg = dict(H.out_degree())
        starts = [v for v in H if outdeg[v] - indeg[v] == 1]
        ends   = [v for v in H if indeg[v] - outdeg[v] == 1]
        mids   = [v for v in H if indeg[v] == 1 and outdeg[v] == 1]
        if len(starts) != 1 or len(ends) != 1 or len(mids) != n - 2:
            return False
        # 시작점에서 후속 간선만 따라가며 모든 노드 1회 방문되는지 확인
        v = starts[0]
        visited = {v}
        while True:
            nxt = [u for u in H.successors(v) if u not in visited]
            if not nxt:
                break
            v = nxt[0]
            visited.add(v)
        return len(visited) == n
    else:
        if not nx.is_connected(H):
            return False
        n, m = H.number_of_nodes(), H.number_of_edges()
        if n == 1:
            return True
        if m != n - 1:
            return False
        degs = dict(H.degree())
        deg1 = sum(1 for d in degs.values() if d == 1)
        deg2 = sum(1 for d in degs.values() if d == 2)
        return deg1 == 2 and deg2 == n - 2

# common_utils.py (상단 유틸 함수들과 같은 위치에 추가)
def neighbors_dir(G, u):
    """방향 그래프면 successors ∪ predecessors, 무향이면 neighbors."""
    if not G.is_directed():
        return list(G.neighbors(u))
    return list(set(G.successors(u)) | set(G.predecessors(u)))


def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        # frontier = list(set(graph.neighbors(start_node)) - set(neigh)) # 기존 이웃 호출 코드
        frontier = list(set(neighbors_dir(graph, start_node)) - set(neigh)) # DiGraph에서는 successors ∪ predecessors 로 이웃 호출 교체
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            # frontier += list(graph.neighbors(new_node)) # 기존 이웃 호출 코드
            frontier += list(neighbors_dir(graph, new_node)) # 수정된 코드
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh

cached_masks = None
def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    #v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    #v = [np.sum(v) for mask in cached_masks]
    return v

def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=int)
        for n in g.nodes:
            # newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0)) # 기존 방향성 방향 안 하는 코드
            # ↓↓ out-이웃과 in-이웃을 구분 가중합(간단히 out=1, in=2)으로 섞어 방향을 부호화
            if g.is_directed():
                succ = list(g.successors(n))
                pred = list(g.predecessors(n))
                agg = np.sum(vecs[succ], axis=0) + 2 * np.sum(vecs[pred], axis=0) + vecs[n]
            else:
                agg = np.sum(vecs[neighbors_dir(g, n) + [n]], axis=0)
            newvecs[n] = vec_hash(agg)
            # ↑↑ 여기까지가 위 newvecs[n] 대체 코드
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))

def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
            progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size: total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts

def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
        node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        # neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
        #     not in sg and nbr not in old_v_ext] # 기존 이웃 확장 코드
        neighbors = [nbr for nbr in neighbors_dir(G, w) if nbr > node_id and nbr not in sg and nbr not in old_v_ext] # 방향 그래프에서도 확장 후보는 succ∪pred 기준
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            #if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
            node_anchored)
        sg.remove(w)

def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
    node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    #for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        #bads, t = 0, 0
        #for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache

def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
            help='Optimizer weight decay.')

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

def batch_nx_graphs(graphs, anchors=None):
    #motifs_batch = [pyg_utils.from_networkx(
    #    nx.convert_node_labels_to_integers(graph)) for graph in graphs]
    #loader = DataLoader(motifs_batch, batch_size=len(motifs_batch))
    #for b in loader: batch = b
    augmenter = feature_preprocess.FeatureAugment()
    
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    batch = batch.to(get_device())
    return batch
