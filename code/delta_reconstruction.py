"""N/M/Delta-only inverse search. No dataset or reference graph is read here.

The public, five-field file interface lives in method.py. This module contains
exact local motif updates, a configuration-model sampler, and a nested search.
All motif coordinates retain their catalogue identity (they are never sorted
by abundance). Repeated hyperedges are distinct, as in the mining code.
"""
import math
import time
from functools import lru_cache
from itertools import permutations

import numba as nb
import numpy as np


@lru_cache(maxsize=8)
def lookup_tables(keys):
    pair = np.full(7 ** 3, -1, dtype=np.int32)
    triple = np.full(7 ** 7, -1, dtype=np.int32)
    for index, key in enumerate(keys):
        r, atom = key[0], key[1:]
        for perm in permutations(range(r)):
            translated = [0] * len(atom)
            for mask in range(1, 1 << r):
                new_mask = sum(1 << perm[j] for j in range(r) if mask & (1 << j))
                translated[new_mask - 1] = atom[mask - 1]
            code = sum(value * 7 ** j for j, value in enumerate(translated))
            (pair if r == 2 else triple)[code] = index
    return pair, triple


def pack(edges):
    sizes = np.asarray([len(edge) for edge in edges], dtype=np.int64)
    table = np.full((len(edges), 6), -1, dtype=np.int64)
    for i, edge in enumerate(edges):
        table[i, :len(edge)] = edge
    return table, sizes


def unpack(table, sizes):
    return [tuple(sorted(map(int, table[i, :size]))) for i, size in enumerate(sizes)]


@nb.njit(cache=True)
def _overlap(a, sa, b, sb):
    total = 0
    for i in range(sa):
        for j in range(sb):
            if a[i] == b[j]:
                total += 1
                break
    return total


@nb.njit(cache=True)
def overlaps(table, sizes):
    m = len(sizes)
    out = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(i):
            out[i, j] = out[j, i] = _overlap(table[i], sizes[i], table[j], sizes[j])
    return out


@nb.njit(cache=True)
def _triple_id(table, sizes, ov, i, j, k, lut):
    ab, ac, bc = int(ov[i, j]), int(ov[i, k]), int(ov[j, k])
    if (ab > 0) + (ac > 0) + (bc > 0) < 2:
        return -1
    shared = 0
    if ab and ac and bc:
        for a in range(sizes[i]):
            node = table[i, a]
            in_b, in_c = False, False
            for b in range(sizes[j]):
                if table[j, b] == node:
                    in_b = True
                    break
            if in_b:
                for c in range(sizes[k]):
                    if table[k, c] == node:
                        in_c = True
                        break
            if in_b and in_c:
                shared += 1
    code = (sizes[i] - ab - ac + shared
            + 7 * (sizes[j] - ab - bc + shared)
            + 49 * (ab - shared)
            + 343 * (sizes[k] - ac - bc + shared)
            + 2401 * (ac - shared) + 16807 * (bc - shared)
            + 117649 * shared)
    return lut[code]


@nb.njit(cache=True)
def count_all(table, sizes, ov, pair, triple, dimension, max_r):
    out = np.zeros(dimension, dtype=np.float64)
    m = len(sizes)
    for i in range(m):
        for j in range(i + 1, m):
            ab = int(ov[i, j])
            if ab:
                out[pair[sizes[i] - ab + 7 * (sizes[j] - ab) + 49 * ab]] += 1
    if max_r == 3:
        # Enumerate connected wedges, not all C(M,3) triples. A path has
        # one center; a triangle is counted only at its smallest edge index.
        for i in range(m):
            neighbors = np.where(ov[i] > 0)[0]
            for a in range(len(neighbors)):
                j = neighbors[a]
                for b in range(a + 1, len(neighbors)):
                    k = neighbors[b]
                    if ov[j, k] > 0 and (j < i or k < i):
                        continue
                    motif = _triple_id(table, sizes, ov, i, j, k, triple)
                    out[motif] += 1
    return out


@nb.njit(cache=True)
def affected_counts(table, sizes, ov, changed, pair, triple, dimension, max_r):
    """Each pair/triple touching a modified edge is counted exactly once."""
    out = np.zeros(dimension, dtype=np.float64)
    m = len(sizes)
    marked = np.zeros(m, dtype=np.bool_)
    for i in changed:
        marked[i] = True
    for i in changed:
        for j in range(m):
            if j == i or (marked[j] and j < i):
                continue
            ab = int(ov[i, j])
            if ab:
                out[pair[sizes[i] - ab + 7 * (sizes[j] - ab) + 49 * ab]] += 1
        if max_r == 3:
            neighbors = np.where(ov[i] > 0)[0]
            # i is the center (includes triangles).
            for a in range(len(neighbors)):
                j = neighbors[a]
                if marked[j] and j < i:
                    continue
                for b in range(a + 1, len(neighbors)):
                    k = neighbors[b]
                    if marked[k] and k < i:
                        continue
                    motif = _triple_id(table, sizes, ov, i, j, k, triple)
                    out[motif] += 1
            # i is an endpoint of a path; j is its unique center.
            for j in neighbors:
                if marked[j] and j < i:
                    continue
                for k in range(m):
                    if k == i or ov[j,k] == 0 or ov[i,k] > 0 or (marked[k] and k < i):
                        continue
                    motif = _triple_id(table, sizes, ov, i, j, k, triple)
                    out[motif] += 1
    return out


@nb.njit(cache=True)
def refresh_overlaps(table, sizes, ov, changed):
    for i in changed:
        for j in range(len(sizes)):
            if i != j:
                ov[i, j] = ov[j, i] = _overlap(table[i], sizes[i], table[j], sizes[j])


def exact_counts(edges, keys, max_r):
    table, sizes = pack(edges)
    pair, triple = lookup_tables(tuple(keys))
    return count_all(table, sizes, overlaps(table, sizes), pair, triple, len(keys), max_r)


@nb.njit(cache=True)
def curveball(table, sizes, node_num, trades, seed):
    """Node-side Hypercurveball with fixed row degrees and labelled edge sizes.

    The transition is the same uniform repartition of exclusive incidences as
    method._func_hm_curveball_trade. The representation avoids Python sets.
    """
    np.random.seed(seed)
    result = table.copy()
    m = len(sizes)
    incidence = np.zeros((node_num, m), dtype=np.bool_)
    node_edges = np.empty((node_num, m), dtype=np.int32)
    degree = np.zeros(node_num, dtype=np.int64)
    for edge in range(m):
        for pos in range(sizes[edge]):
            node = result[edge, pos]
            incidence[node, edge] = True
            node_edges[node, degree[node]] = edge
            degree[node] += 1
    pool = np.empty(2 * m, dtype=np.int32)
    common = np.empty(m, dtype=np.int32)
    for _ in range(trades):
        left = np.random.randint(node_num)
        right = np.random.randint(node_num - 1)
        if right >= left:
            right += 1
        n_pool, n_common = 0, 0
        for p in range(degree[left]):
            edge = node_edges[left, p]
            if incidence[right, edge]:
                common[n_common] = edge
                n_common += 1
            else:
                pool[n_pool] = edge
                n_pool += 1
        n_left = n_pool
        for p in range(degree[right]):
            edge = node_edges[right, p]
            if not incidence[left, edge]:
                pool[n_pool] = edge
                n_pool += 1
        if n_pool == 0:
            continue
        for p in range(n_pool - 1, 0, -1):
            q = np.random.randint(p + 1)
            pool[p], pool[q] = pool[q], pool[p]
        for p in range(n_common):
            node_edges[left, p] = common[p]
            node_edges[right, p] = common[p]
        for p in range(n_pool):
            edge = pool[p]
            dest = left if p < n_left else right
            source = right if p < n_left else left
            offset = n_common + (p if p < n_left else p - n_left)
            node_edges[dest, offset] = edge
            if not incidence[dest, edge]:
                for pos in range(sizes[edge]):
                    if result[edge, pos] == source:
                        result[edge, pos] = dest
                        break
                incidence[source, edge] = False
                incidence[dest, edge] = True
    return result


def null_mean(edges, node_num, keys, max_r, null_num, burn, seed, deadline=None):
    table, sizes = pack(edges)
    pair, triple = lookup_tables(tuple(keys))
    rng = np.random.default_rng(seed)
    total = np.zeros(len(keys))
    for _ in range(null_num):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Null estimation reached the search time budget.')
        randomized = curveball(table, sizes, node_num, burn, int(rng.integers(2 ** 31)))
        total += count_all(randomized, sizes, overlaps(randomized, sizes),
                           pair, triple, len(keys), max_r)
    return total / null_num


def delta(counts, mean):
    return (counts - mean) / (counts + mean + 4.)


def loss_weights(target, keys):
    """Equal order weights; 70% weight on target amplitude, 30% on all bins.

    Weights are fixed before search, so creating spurious nonzero motifs is
    penalized and neither zero padding nor an evolving active mask can help.
    """
    weights = np.zeros(len(keys))
    order = np.asarray([key[0] for key in keys])
    orders = np.unique(order)
    for r in orders:
        mask = order == r
        amplitude = np.abs(target[mask])
        if amplitude.sum() > 1e-12:
            weights[mask] = (.3 / mask.sum() + .7 * amplitude / amplitude.sum()) / len(orders)
        else:
            weights[mask] = 1. / (mask.sum() * len(orders))
    return weights


@nb.njit(cache=True)
def score(counts, mean, target, weights):
    out = 0.
    for i in range(len(target)):
        difference = (counts[i] - mean[i]) / (counts[i] + mean[i] + 4.) - target[i]
        out += weights[i] * difference * difference
    return out


def cardinality_prior(target, keys, max_v):
    """Only Delta is used; this is a proposal distribution, not an inferred truth."""
    support = np.zeros(max_v + 1)
    groups = {}
    for value, key in zip(target, keys):
        if abs(value) < .03:
            continue
        r, atom = key[0], key[1:]
        sizes = tuple(sorted(sum(atom[mask-1] for mask in range(1, 1 << r)
                                 if mask & (1 << j)) for j in range(r)))
        # Avoid giving a size inventory extra weight merely because it has
        # many possible three-edge motifs in the catalogue.
        groups[sizes] = max(groups.get(sizes, 0.), abs(value))
    for sizes, value in groups.items():
        for size in sizes:
            support[size] += value / len(sizes)
    support[2:] += max(.05, support.sum() * .01)
    return support[2:] / support[2:].sum()


def seed_graph(node_num, edge_num, max_v, rng, probabilities=None, locality=0.):
    """Random connected graph with an independently sampled size inventory.

    A rolling recent-node pool generates clustered references/seeds. It never
    receives the source graph, source degrees or source cardinalities.
    """
    cap = min(node_num, max_v)
    probability = None if probabilities is None else np.array(probabilities[:cap-1], copy=True)
    if probability is not None:
        probability /= probability.sum()
    sizes = rng.choice(np.arange(2, cap+1), edge_num, p=probability)
    # Connectivity needs at least N+M-1 incidences.
    while int(sizes.sum()) < node_num + edge_num - 1:
        available = np.flatnonzero(sizes < cap)
        if not len(available):
            raise ValueError('N/M cannot support a connected graph.')
        count = min(node_num + edge_num - 1 - int(sizes.sum()), len(available))
        sizes[rng.choice(available, count, replace=False)] += 1
    labels = rng.permutation(node_num).tolist()
    visited, position, edges = [labels[0]], 1, []
    remaining_capacity = int(np.sum(sizes-1))
    for size in sizes:
        remaining_capacity -= int(size)-1
        left = node_num-position
        low = max(0, left-remaining_capacity, int(size)-len(visited))
        high = min(int(size)-1, left)
        # Introduce nodes throughout the graph rather than using all nodes
        # immediately and leaving a completely random dense tail.
        expected = left * (int(size)-1) / max(1, remaining_capacity+int(size)-1)
        added = int(np.clip(rng.binomial(high, min(1., expected/max(1,high))), low, high))
        old_num = int(size)-added
        pool = visited
        if locality and rng.random() < locality:
            width = max(old_num, min(len(visited), int(rng.integers(cap, max(cap+1, 3*cap)))))
            pool = visited[-width:]
        old = rng.choice(pool, old_num, replace=False).tolist()
        fresh = labels[position:position+added]
        edges.append(tuple(sorted(old+fresh)))
        visited.extend(fresh)
        position += added
    return edges


@nb.njit(cache=True)
def _connected(table, sizes, node_num):
    parent = np.arange(node_num)
    used = np.zeros(node_num, dtype=np.bool_)
    for i in range(len(sizes)):
        root = table[i, 0]
        while parent[root] != root:
            root = parent[root]
        for p in range(sizes[i]):
            node = table[i, p]
            used[node] = True
            while parent[node] != node:
                node = parent[node]
            parent[node] = root
    first = 0
    while parent[first] != first:
        first = parent[first]
    for node in range(node_num):
        if not used[node]:
            return False
        while parent[node] != node:
            node = parent[node]
        if node != first:
            return False
    return True


@nb.njit(cache=True)
def local_search(table, sizes, mean, target, weights, pair, triple, max_r,
                 node_num, steps, temperature, seed):
    """Metropolis incidence swaps with exact incremental motif counts."""
    np.random.seed(seed)
    table = table.copy()
    ov = overlaps(table, sizes)
    counts = count_all(table, sizes, ov, pair, triple, len(target), max_r)
    current_loss = score(counts, mean, target, weights)
    best_loss, best = current_loss, table.copy()
    m = len(sizes)
    if m < 2:
        return best, best_loss
    for step in range(steps):
        i = np.random.randint(m)
        j = np.random.randint(m-1)
        if j >= i:
            j += 1
        # Half the proposals join the neighborhood of i, the rest explore
        # globally. Both preserve degrees and each labelled edge's size.
        if np.random.random() < .5:
            anchor = np.random.randint(m)
            if ov[i, anchor] > 0:
                node = table[anchor, np.random.randint(sizes[anchor])]
                start = np.random.randint(m)
                for offset in range(m):
                    candidate = (start+offset) % m
                    if candidate == i:
                        continue
                    found = False
                    for p in range(sizes[candidate]):
                        if table[candidate, p] == node:
                            j = candidate
                            found = True
                            break
                    if found:
                        break
        a, b = np.random.randint(sizes[i]), np.random.randint(sizes[j])
        left, right = table[i, a], table[j, b]
        if left == right:
            continue
        valid = True
        for p in range(sizes[i]):
            if table[i, p] == right:
                valid = False
        for p in range(sizes[j]):
            if table[j, p] == left:
                valid = False
        if not valid:
            continue
        changed = np.array([i, j])
        before = affected_counts(table, sizes, ov, changed, pair, triple, len(target), max_r)
        table[i, a], table[j, b] = right, left
        refresh_overlaps(table, sizes, ov, changed)
        after = affected_counts(table, sizes, ov, changed, pair, triple, len(target), max_r)
        trial_counts = counts + after - before
        trial_loss = score(trial_counts, mean, target, weights)
        accept = trial_loss <= current_loss
        if not accept and temperature > 0:
            accept = np.random.random() < math.exp(min(0., (current_loss-trial_loss)/temperature))
        if accept:
            accept = _connected(table, sizes, node_num)
        if accept:
            counts, current_loss = trial_counts, trial_loss
            if current_loss < best_loss:
                best, best_loss = table.copy(), current_loss
        else:
            table[i, a], table[j, b] = left, right
            refresh_overlaps(table, sizes, ov, changed)
    return best, best_loss


def margin_proposal(edges, node_num, max_v, rng, probabilities):
    table, sizes = pack(edges)
    for _ in range(int(rng.integers(1, max(3, len(edges)//30+1)))):
        i = int(rng.integers(len(edges)))
        edge = set(table[i, :sizes[i]])
        choice = rng.random()
        available = np.asarray([v for v in range(node_num) if v not in edge])
        if choice < .4 and len(available):
            edge.remove(int(rng.choice(sorted(edge))))
            edge.add(int(rng.choice(available)))
        elif choice < .75:
            desired = int(rng.choice(np.arange(2, max_v+1), p=probabilities))
            if desired > len(edge) and len(available) and len(edge) < max_v:
                edge.add(int(rng.choice(available)))
            elif desired < len(edge) and len(edge) > 2:
                edge.remove(int(rng.choice(sorted(edge))))
        elif len(available) and len(edge) < max_v:
            edge.add(int(rng.choice(available)))
        elif len(edge) > 2:
            edge.remove(int(rng.choice(sorted(edge))))
        table[i] = -1
        sizes[i] = len(edge)
        table[i, :sizes[i]] = sorted(edge)
    if _connected(table, sizes, node_num):
        return unpack(table, sizes)
    return None


def motif_proposal(edges, node_num, keys, residual, rng):
    """Plant an underrepresented positive motif, without copying source edges.

    Vertices used only by the replaced edges are retained, so a template cannot
    improve its score simply by dropping hard-to-cover nodes. Connectivity is
    checked again after the complete replacement.
    """
    probabilities = np.maximum(residual, 0.) ** 2
    if probabilities.sum() <= 1e-15:
        return None
    probabilities /= probabilities.sum()
    degree = np.bincount([node for edge in edges for node in edge], minlength=node_num)
    for _ in range(6):
        key = keys[int(rng.choice(len(keys), p=probabilities))]
        r, atom = key[0], key[1:]
        if r > len(edges) or sum(atom) > node_num:
            continue
        anchor = int(rng.integers(len(edges)))
        neighborhood = [i for i, edge in enumerate(edges)
                        if i != anchor and set(edge).intersection(edges[anchor])]
        if len(neighborhood) < r-1:
            continue
        selected = [anchor] + rng.choice(neighborhood, r-1, replace=False).tolist()
        local_degree = np.bincount([node for i in selected for node in edges[i]], minlength=node_num)
        mandatory = np.flatnonzero((local_degree > 0) & (local_degree == degree)).tolist()
        union_size = sum(atom)
        if len(mandatory) > union_size:
            continue
        old = [node for node in np.flatnonzero(local_degree) if node not in mandatory]
        rng.shuffle(old)
        labels = mandatory + old[:union_size-len(mandatory)]
        if len(labels) < union_size:
            available = [node for node in range(node_num) if node not in labels]
            labels += rng.choice(available, union_size-len(labels), replace=False).tolist()
        rng.shuffle(labels)
        planted = [[] for _ in range(r)]
        position = 0
        for mask, value in enumerate(atom, start=1):
            for j in range(r):
                if mask & (1 << j):
                    planted[j].extend(labels[position:position+value])
            position += value
        candidate = list(edges)
        for i, edge in zip(selected, planted):
            candidate[i] = tuple(sorted(edge))
        table, sizes = pack(candidate)
        if _connected(table, sizes, node_num):
            return candidate
    return None


def _search_legacy(initial, node_num, target, keys, max_r, max_v, seed, max_iter,
           max_seconds, tolerance, search_null_num, null_estimator, verbose=False):
    """Alternating margin search and fixed-margin refinement, with fresh ranking.

    null_estimator(edges, samples, seed, deadline) must generate actual nulls.
    The final reporting null ensemble is deliberately outside this search.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    deadline = start + max_seconds
    keys = tuple(keys)
    pair, triple = lookup_tables(keys)
    weights = loss_weights(target, keys)
    prior = cardinality_prior(target, keys, max_v)
    best, current = list(initial), list(initial)
    best_loss = np.inf
    archive, history = [], []
    used, epochs = 0, 0

    def evaluate(edges, samples=search_null_num):
        counts = exact_counts(edges, keys, max_r)
        mean = null_estimator(edges, samples, int(rng.integers(2**31)), deadline)
        return score(counts, mean, target, weights), mean, counts

    def remember(edges, loss):
        nonlocal best, best_loss
        identity = tuple(sorted(edges))
        if all(identity != entry[2] for entry in archive):
            archive.append((float(loss), list(edges), identity))
            archive.sort(key=lambda entry: entry[0])
            del archive[6:]
        if loss < best_loss:
            best, best_loss = list(edges), float(loss)

    try:
        current_loss, mean, counts = evaluate(current)
        remember(current, current_loss)
        history.append({'step': 0, 'weighted_rmse': math.sqrt(best_loss)})
        if max_iter == 0:
            return best, history
        # Multiple size inventories and clustering levels, inferred from Delta
        # only. Charge each seed against the caller's candidate budget.
        for trial in range(min(10 if len(initial) <= 100 else 4, max_iter//50)):
            if time.perf_counter() >= deadline:
                break
            candidate = seed_graph(node_num, len(initial), max_v, rng, prior,
                                   locality=(0., .65, .95)[trial % 3])
            loss, trial_mean, trial_counts = evaluate(candidate)
            used += 1
            remember(candidate, loss)
            if loss < current_loss:
                current, current_loss, mean, counts = candidate, loss, trial_mean, trial_counts
        while used < max_iter and time.perf_counter() < start + .85*max_seconds:
            epochs += 1
            table, sizes = pack(current)
            block = min(160, max_iter-used)
            # Temperature scales with the mean improvement of a single swap;
            # the returned incumbent is always the best visited graph.
            temperature = .00002 * max(.05, 1.-used/max(1,max_iter))
            table, _ = local_search(table, sizes, mean, target, weights, pair,
                                     triple, max_r, node_num, block, temperature,
                                     int(rng.integers(2**31)))
            used += block
            current = unpack(table, sizes)
            counts = exact_counts(current, keys, max_r)
            current_loss = score(counts, mean, target, weights)
            remember(current, current_loss)
            history.append({'step': used, 'weighted_rmse': math.sqrt(best_loss)})
            if verbose and (epochs == 1 or epochs % 5 == 0):
                print(f'  Delta search: {used}/{max_iter}, weighted RMSE={math.sqrt(best_loss):.5f}, '
                      f'{time.perf_counter()-start:.1f}s', flush=True)
            if np.max(np.abs(delta(counts, mean)-target)) <= tolerance:
                break
            if used >= max_iter:
                break
            # Old-null scores only screen proposals. No changed-margin graph
            # is accepted or archived before evaluating its OWN null ensemble.
            proposals = []
            residual = np.where(target > .1, target-delta(counts, mean), 0.)
            for proposal_index in range(min(18, max_iter-used)):
                if time.perf_counter() >= deadline:
                    break
                if proposal_index % 3 == 0:
                    candidate = motif_proposal(current, node_num, keys, residual, rng)
                else:
                    candidate = margin_proposal(current, node_num, max_v, rng, prior)
                used += 1
                if candidate is not None:
                    c = exact_counts(candidate, keys, max_r)
                    proposals.append((score(c, mean, target, weights), candidate))
            proposals.sort(key=lambda item:item[0])
            for _, candidate in proposals[:2]:
                loss, trial_mean, trial_counts = evaluate(candidate)
                remember(candidate, loss)
                # Small exploratory moves can cross a local basin, but cannot
                # overwrite the incumbent's best-score archive.
                if loss < current_loss or rng.random() < .08:
                    current, current_loss, mean, counts = candidate, loss, trial_mean, trial_counts
            if epochs % 5 == 0:
                current = list(best)
                current_loss, mean, counts = evaluate(current, min(4*search_null_num, 40))
                # Refresh the incumbent estimate, rather than keeping an
                # unrealistically optimistic score from a noisy null draw.
                archive = [entry for entry in archive if entry[2] != tuple(sorted(current))]
                best_loss = current_loss
                remember(current, current_loss)
    except TimeoutError:
        pass
    # Independent search-validation stream. Compare all finalists using the
    # same sample count; final reporting gets yet another independent stream.
    ranked = []
    for _, candidate, _ in archive[:4]:
        if time.perf_counter() >= deadline:
            break
        try:
            loss, _, _ = evaluate(candidate, max(16, 4*search_null_num))
            ranked.append((loss, candidate))
        except TimeoutError:
            break
    if ranked:
        best_loss, best = min(ranked, key=lambda item:item[0])
    if verbose:
        print(f'  Delta search complete: proposals={used}, epochs={epochs}, '
              f'weighted RMSE estimate={math.sqrt(best_loss):.5f}', flush=True)
    return list(best), history


def search(initial, node_num, target, keys, max_r, max_v, seed, max_iter,
           max_seconds, tolerance, search_null_num, null_estimator, verbose=False):
    """Use enriched search only after target demand and seed advantage checks.

    The established solver remains the fallback. Both enhanced stages share
    the original time/proposal budget; final reporting nulls remain outside
    this routine, in the unchanged public generation entry point.
    """
    if max_iter < 200:
        return _search_legacy(initial, node_num, target, keys, max_r, max_v, seed,
                              max_iter, max_seconds, tolerance, search_null_num,
                              null_estimator, verbose)
    from delta_reconstruction_optimizer import search as enhanced_search
    return enhanced_search(initial, node_num, target, keys, max_r, max_v, seed,
                            max_iter, max_seconds, tolerance, search_null_num,
                            null_estimator, verbose, legacy_search=_search_legacy)
