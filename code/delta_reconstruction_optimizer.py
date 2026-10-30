"""Target-driven seed comparison and population refinement for N/M/Delta.

No reference incidence structure, degrees or edge sizes are supplied.
Changed-margin candidates always receive their own configuration null;
method.py computes another independent null ensemble for final reporting.
"""
import math
import time
import numpy as np
import numba as nb
import delta_reconstruction as k


def copy_seed(n, m, cap, rng, probabilities, copying, locality, heterogeneity):
    """Connected growth with tunable overlap, vertex preference and size mix."""
    cap = min(n, cap)
    p = np.array(probabilities[:cap-1], copy=True)
    p /= p.sum()
    sizes = rng.choice(np.arange(2, cap+1), m, p=p)
    while sizes.sum() < n+m-1:
        can = np.flatnonzero(sizes < cap)
        if not len(can):
            raise ValueError('N/M cannot support a connected graph.')
        take = min(n+m-1-int(sizes.sum()), len(can))
        sizes[rng.choice(can, take, replace=False)] += 1
    labels = rng.permutation(n).tolist()
    visited = [labels[0]]
    position = 1
    edges = []
    degrees = np.zeros(n)
    remaining = int(np.sum(sizes-1))
    for s in sizes:
        remaining -= int(s)-1
        left = n-position
        low = max(0, left-remaining, int(s)-len(visited))
        high = min(int(s)-1, left)
        expected = left*(int(s)-1)/max(1, remaining+int(s)-1)
        added = int(np.clip(rng.binomial(high, min(1., expected/max(1, high))), low, high))
        wanted = int(s)-added
        chosen = []
        if edges and rng.random() < copying:
            window = max(1, int((1.-locality)*len(edges)))
            anchor = edges[int(rng.integers(max(0, len(edges)-window), len(edges)))]
            take = min(wanted, len(anchor))
            chosen = rng.choice(anchor, take, replace=False).tolist()
        if len(chosen) < wanted:
            pool = [v for v in visited if v not in chosen]
            weights = (1.+degrees[pool])**heterogeneity
            if rng.random() < locality:
                recent = set(visited[-max(cap, 3*cap):])
                weights *= np.array([10. if v in recent else 1. for v in pool])
            weights /= weights.sum()
            chosen += rng.choice(pool, wanted-len(chosen), replace=False, p=weights).tolist()
        fresh = labels[position:position+added]
        edge = tuple(sorted(chosen+fresh))
        edges.append(edge)
        degrees[list(edge)] += 1
        visited += fresh
        position += added
    assert position == n
    return edges


@nb.njit(cache=True)
def descent(table, sizes, mean, target, weights, pair, triple, max_r, n, steps, batch, seed, flexible):
    """Best improving move in sampled neighborhoods; no annealing acceptance.

    Flexible moves use a frozen-null surrogate only. Their batch is accepted
    outside this routine after recomputing the graph's own null expectation.
    """
    np.random.seed(seed)
    table = table.copy()
    sizes = sizes.copy()
    ov = k.overlaps(table, sizes)
    counts = k.count_all(table, sizes, ov, pair, triple, len(target), max_r)
    loss = k.score(counts, mean, target, weights)
    m = len(sizes)
    used = 0
    accepted = 0
    idle = 0
    degree = np.zeros(n, dtype=np.int64)
    for i in range(m):
        for p in range(sizes[i]):
            degree[table[i, p]] += 1
    while used < steps:
        best_loss = loss
        best_table = table.copy()
        best_sizes = sizes.copy()
        best_counts = counts.copy()
        changed_best = np.empty(0, dtype=np.int64)
        for _ in range(min(batch, steps-used)):
            used += 1
            i = np.random.randint(m)
            a = np.random.randint(sizes[i])
            left = table[i, a]
            old_i = table[i].copy()
            old_si = sizes[i]
            j = -1
            b = -1
            action = np.random.random() if flexible else 0.
            if action < .35 and m > 1:
                j = np.random.randint(m-1)
                if j >= i:
                    j += 1
                b = np.random.randint(sizes[j])
                right = table[j, b]
                if right == left:
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
                before = k.affected_counts(table, sizes, ov, changed,
                                           pair, triple, len(target), max_r)
                table[i, a], table[j, b] = right, left
            elif action < .83:
                if degree[left] <= 1:
                    continue
                # A mixture of neighbor nodes and global nodes explores both
                # overlap formation and destruction, without reference edges.
                if np.random.random() < .65:
                    anchor = np.random.randint(m)
                    neighbors = np.where(ov[i] > 0)[0]
                    if len(neighbors) > 0:
                        anchor = neighbors[np.random.randint(len(neighbors))]
                    right = table[anchor, np.random.randint(sizes[anchor])]
                else:
                    right = np.random.randint(n)
                valid = True
                for p in range(sizes[i]):
                    if table[i, p] == right:
                        valid = False
                if not valid:
                    continue
                changed = np.array([i])
                before = k.affected_counts(table, sizes, ov, changed,
                                           pair, triple, len(target), max_r)
                table[i, a] = right
            elif action < .915:
                if sizes[i] >= table.shape[1] or sizes[i] >= n:
                    continue
                right = np.random.randint(n)
                valid = True
                for p in range(sizes[i]):
                    if table[i, p] == right:
                        valid = False
                if not valid:
                    continue
                changed = np.array([i])
                before = k.affected_counts(table, sizes, ov, changed,
                                           pair, triple, len(target), max_r)
                table[i, sizes[i]] = right
                sizes[i] += 1
            else:
                if sizes[i] <= 2 or degree[left] <= 1:
                    continue
                changed = np.array([i])
                before = k.affected_counts(table, sizes, ov, changed,
                                           pair, triple, len(target), max_r)
                sizes[i] -= 1
                table[i, a] = table[i, sizes[i]]
                table[i, sizes[i]] = -1
            k.refresh_overlaps(table, sizes, ov, changed)
            after = k.affected_counts(table, sizes, ov, changed, pair, triple, len(target), max_r)
            c = counts+after-before
            value = k.score(c, mean, target, weights)
            if value < best_loss-1e-14 and k._connected(table, sizes, n):
                best_loss = value
                best_table = table.copy()
                best_sizes = sizes.copy()
                best_counts = c.copy()
                changed_best = changed.copy()
            table[i] = old_i
            sizes[i] = old_si
            if j >= 0:
                table[j, b] = right
            k.refresh_overlaps(table, sizes, ov, changed)
        if best_loss < loss:
            table = best_table
            sizes = best_sizes
            counts = best_counts
            loss = best_loss
            k.refresh_overlaps(table, sizes, ov, changed_best)
            degree[:] = 0
            for i in range(m):
                for p in range(sizes[i]):
                    degree[table[i, p]] += 1
            accepted += 1
            idle = 0
        else:
            idle += 1
            if idle >= 5:
                break
    return table, sizes, loss, used, accepted


def diversity(edges, n, max_v):
    sizes = np.bincount(list(map(len, edges)), minlength=max_v+1)[2:]/len(edges)
    deg = np.bincount([v for e in edges for v in e], minlength=n)
    return np.r_[sizes, np.quantile(deg, [.25, .5, .75, .9, 1.])/max(1., deg.mean())*.1]


def _population_search(initial, n, target, keys, max_r, max_v, seed, max_iter, max_seconds, tolerance,
                       search_null_num, null_estimator, verbose=False):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    deadline = start+max_seconds
    weights = k.loss_weights(target, keys)
    pair, triple = k.lookup_tables(tuple(keys))
    prior = k.cardinality_prior(target, keys, max_v)
    population = []
    archive = []
    history = []
    used = 0
    epochs = 0
    best = list(initial)
    # Full evaluation always generates actual fixed-margin null graphs.

    def evaluate(edges, samples):
        c = k.exact_counts(edges, keys, max_r)
        mu = null_estimator(edges, samples, int(rng.integers(2**31)), deadline)
        value = k.score(c, mu, target, weights)
        return dict(edges=list(edges), loss=float(value), mean=mu, counts=c,
                    feature=diversity(edges, n, max_v), visits=0, stalls=0)

    def remember(item):
        nonlocal best
        identity = tuple(sorted(item['edges']))
        archive[:] = [x for x in archive if x[2] != identity]
        archive.append((item['loss'], list(item['edges']), identity))
        archive.sort(key=lambda x: x[0])
        del archive[8:]
        best = list(archive[0][1])

    def add(item):
        remember(item)
        for j, other in enumerate(population):
            if np.linalg.norm(other['feature']-item['feature']) < .035:
                if item['loss'] < other['loss']:
                    population[j] = item
                return
        population.append(item)
        population.sort(key=lambda x: x['loss'])
        del population[4:]
    try:
        initial_item = evaluate(initial, search_null_num)
        add(initial_item)
        history.append(dict(stage='initial', loss=initial_item['loss'], used=used))
        if max_iter == 0:
            return best, history
        seed_records = []
        # Stratified generation followed by elite adaptation of the sampling
        # law. Copying can create high-overlap motifs at initialization.
        seed_limit = min(24, max_iter//40)
        old_seeds = 10 if len(initial) <= 100 else 4
        for t in range(seed_limit):
            if time.perf_counter() > start+.25*max_seconds:
                break
            if t < old_seeds:
                params = (0., (0., .65, .95)[t % 3], 0.)
                probabilities = prior
                edges = k.seed_graph(n, len(initial), max_v, rng, probabilities, locality=params[1])
            else:
                if t < old_seeds+8 or not seed_records:
                    params = (float(rng.choice([.25, .6, .9, 1.])), float(
                        rng.uniform(.2, 1.)), float(rng.choice([0., .5, 1., 1.5])))
                    probabilities = rng.dirichlet(
                        np.maximum(.15, prior*float(rng.choice([3., 12., 50.]))))
                else:
                    elite = sorted(seed_records, key=lambda x: x[0])[:4]
                    parent = elite[int(rng.integers(len(elite)))]
                    parameters = np.asarray(parent[1])+rng.normal(0., [.15, .15, .25])
                    parameters[:2] = np.clip(parameters[:2], 0., 1.)
                    parameters[2] = np.clip(parameters[2], 0., 2.)
                    params = tuple(parameters)
                    learned = np.mean([x[2] for x in elite], axis=0)
                    probabilities = rng.dirichlet(np.maximum(.2, 20*(.3*prior+.7*learned)))
                edges = copy_seed(n, len(initial), max_v, rng, probabilities, *params)
            item = evaluate(edges, max(2, search_null_num//2))
            used += 1
            seed_records.append((item['loss'], params, np.bincount(
                list(map(len, edges)), minlength=max_v+1)[2:]/len(edges)))
            add(item)
        # Equal precision before choosing which members to improve.
        population = [evaluate(x['edges'], search_null_num) for x in population]
        population.sort(key=lambda x: x['loss'])
        for item in population:
            remember(item)
        history.append(dict(stage='seeds', loss=population[0]['loss'], used=used))
        if verbose:
            print('SEEDS', [(round(x['loss']**.5, 4), x['feature']
                  [:max_v-1].round(2).tolist()) for x in population], flush=True)
        rewards = np.full(2, 1e-3)
        while used < max_iter and time.perf_counter() < start+.86*max_seconds:
            epochs += 1
            population.sort(key=lambda x: x['loss'])
            leader = population[0]
            if np.max(np.abs(k.delta(leader['counts'], leader['mean']) - target)) <= tolerance:
                break
            # Mostly exploit the best, but periodically polish another basin.
            index = 0 if epochs % 12 else (epochs//12) % len(population)
            parent = population[index]
            table, sizes = k.pack(parent['edges'])
            # Pad width is already 6; limit allowed size through the input
            # array's column count for smaller catalogue configurations.
            table = table[:, :max_v].copy()
            block = min(160, max_iter-used)
            probability = .15+.7*rewards[1]/max(1e-12, rewards.sum())
            operator = (epochs-1) if epochs <= 2 else int(rng.random() < probability)
            op_start = time.perf_counter()
            if operator:
                result, new_sizes, _, spent, accepted = descent(table, sizes, parent['mean'], target, weights,
                                                                pair, triple, max_r, n, block, 4, int(rng.integers(2**31)), True)
            else:
                result, local_loss = k.local_search(table, sizes, parent['mean'], target, weights, pair,
                                                    triple, max_r, n, block, 2e-5*max(.05, 1-used/max_iter), int(rng.integers(2**31)))
                new_sizes = sizes
                spent = block
                accepted = int(local_loss < parent['loss']-1e-14)
            used += spent
            edges = k.unpack(result, new_sizes)
            parent['visits'] += 1
            gain = 0.
            if accepted:
                if operator:
                    child = evaluate(edges, search_null_num)
                else:
                    c = k.exact_counts(edges, keys, max_r)
                    child = dict(edges=edges, loss=float(k.score(c, parent['mean'], target, weights)),
                                 counts=c, mean=parent['mean'], feature=diversity(edges, n, max_v), visits=0, stalls=0)
                if child['loss'] < parent['loss']:
                    gain = (parent['loss']-child['loss'])/max(parent['loss'], 1e-12)
                    population[index] = child
                    remember(child)
                else:
                    parent['stalls'] += 1
            else:
                parent['stalls'] += 1
            reward = gain/max(.01, time.perf_counter()-op_start)
            rewards[operator] = .65*rewards[operator]+.35*reward
            if epochs % 4 == 0:
                # Fine margin changes complement the longer flexible descent:
                # its stale-null batch can miss useful single-edge changes.
                parent = population[0]
                residual = np.where(target > .1, target -
                                    k.delta(parent['counts'], parent['mean']), 0.)
                proposals = []
                for j in range(min(12, max_iter-used)):
                    if j % 3 == 0:
                        proposal = k.motif_proposal(parent['edges'], n, keys, residual, rng)
                    else:
                        proposal = k.margin_proposal(parent['edges'], n, max_v, rng, prior)
                    used += 1
                    if proposal is not None:
                        c = k.exact_counts(proposal, keys, max_r)
                        proposals.append((k.score(c, parent['mean'], target, weights), proposal))
                proposals.sort(key=lambda x: x[0])
                for _, proposal in proposals[:2]:
                    add(evaluate(proposal, search_null_num))
            if epochs % 8 == 0:
                for item in population[:2]:
                    refreshed = evaluate(item['edges'], max(16, 2*search_null_num))
                    item.update(refreshed)
                    remember(item)
            if verbose and (epochs == 1 or epochs % 5 == 0):
                print('OPT', epochs, used, round(
                    min(x['loss'] for x in population)**.5, 5), round(time.perf_counter()-start, 2), flush=True)
            history.append(dict(stage='search', loss=min(x['loss'] for x in population), used=used,
                                operator=int(operator), flexible_probability=float(probability)))
    except TimeoutError:
        pass
    ranked = []
    for _, edges, _ in archive[:4]:
        try:
            candidate = evaluate(edges, max(20, 4*search_null_num))
            ranked.append(candidate)
        except TimeoutError:
            break
    if ranked:
        best = min(ranked, key=lambda x: x['loss'])['edges']
    return list(best), history


def overlap_demand(node_num, edge_num, target, keys, max_v, tolerance=.05):
    """Compare a necessary positive-motif count with an initialization moment.

    The independent uniform-edge reference used here is ONLY a diagnostic of
    an unstructured starting point. It is not the fixed-margin null model used
    anywhere in the Delta objective or in the final evaluation.
    """
    cap = min(node_num, max_v)
    prior = k.cardinality_prior(target, keys, max_v)[:cap-1].copy()
    prior /= prior.sum()
    required = 0
    expected = 0.
    finite = True
    for value, key in zip(target, keys):
        if key[0] != 2 or key[3] < 2:
            continue
        a, b, c = key[1:]
        sa, sb = a+c, b+c
        # An impossible catalogue coordinate cannot guide feasible seeding.
        if a+b+c > node_num or max(sa, sb) > cap:
            continue
        probability = math.comb(sa, c)*math.comb(node_num-sa, b)/math.comb(node_num, sb)
        multiplicity = 1 if a == b else 2
        expected += math.comb(edge_num, 2)*prior[sa-2]*prior[sb-2]*probability*multiplicity
        effective = float(value)-max(0., tolerance)
        if effective >= 1:
            finite = False
        elif effective > 0:
            required += max(0, math.ceil(4*effective/(1-effective)-1e-10))
    ratio = float(required/max(expected, 1e-12))
    return dict(required_pairs=required, unstructured_expected_pairs=float(expected),
                ratio=ratio, enriched=bool(finite and required > expected+1e-12 and edge_num > 1))


def _compare_initializations(initial, n, target, keys, max_r, max_v, seed, max_iter, deadline, estimator, samples=10):
    rng = np.random.default_rng(seed)
    rich_rng = np.random.default_rng(np.random.SeedSequence([seed, 913327]))
    weights = k.loss_weights(target, keys)
    prior = k.cardinality_prior(target, keys, max_v)
    used = 0
    old = None
    rich = None
    validated = False

    def evaluate(edges, samples, random):
        count = k.exact_counts(edges, keys, max_r)
        mean = estimator(edges, samples, int(random.integers(2**31)), deadline)
        return (float(k.score(count, mean, target, weights)), list(edges))
    try:
        old = evaluate(initial, samples, rng)
        for t in range(min(10 if len(initial) <= 100 else 4, max_iter//50)):
            graph = k.seed_graph(n, len(initial), max_v, rng, prior, locality=(0., .65, .95)[t % 3])
            used += 1
            item = evaluate(graph, samples, rng)
            if item[0] < old[0]:
                old = item
        for t in range(min(12, max_iter//100)):
            distribution = prior if t % 4 == 0 else .3*prior+.7 * \
                rich_rng.dirichlet(np.maximum(.2, 20*prior))
            graph = copy_seed(n, len(initial), max_v, rich_rng, distribution,
                              (.25, .6, .9, 1.)[t % 4], float(rich_rng.uniform(.4, 1.)), (0., .5, 1.)[(t//4) % 3])
            used += 1
            item = evaluate(graph, max(2, samples//2), rich_rng)
            if rich is None or item[0] < rich[0]:
                rich = item
        # Equal precision on independent search streams before a switch.
        if rich is not None and rich[0] < .81*old[0]:
            checked_old = evaluate(old[1], max(20, 2*samples), rich_rng)
            checked_rich = evaluate(rich[1], max(20, 2*samples), rich_rng)
            old, rich = checked_old, checked_rich
            validated = True
    except TimeoutError:
        pass
    switch = bool(validated and rich[0] < .81*old[0])
    return (rich[1] if switch else initial), dict(
        stage='initialization_comparison', used=used, switch=switch, validated=validated,
        old_seed_rmse=None if old is None else math.sqrt(old[0]),
        rich_seed_rmse=None if rich is None else math.sqrt(rich[0]))


def _explore_and_refine(initial, node_num, target, keys, max_r, max_v, seed, max_iter, max_seconds,
                        tolerance, search_null_num, null_estimator, verbose=False, *, legacy_search):
    info = overlap_demand(node_num, len(initial), target, keys, max_v, tolerance)
    if verbose:
        print('OVERLAP DEMAND', info, flush=True)
    args = (node_num, target, keys, max_r, max_v)
    if not info['enriched'] or max_iter < 200:
        return legacy_search(initial, *args, seed, max_iter, max_seconds, tolerance,
                             search_null_num, null_estimator, verbose)
    started = time.perf_counter()
    exploration = max_iter//2
    graph, history = _population_search(initial, *args, seed, exploration, .55*max_seconds,
                                        tolerance, search_null_num, null_estimator, verbose)
    # Reserve the full first-stage allowance: a timeout can occur after a
    # proposal but before the history checkpoint, so its last row can lag.
    spent = exploration
    remaining_time = max_seconds-(time.perf_counter()-started)
    remaining_steps = max_iter-spent
    if remaining_time <= 0 or remaining_steps <= 0:
        return graph, history
    refine_seed = int(np.random.SeedSequence([seed, 872731]).generate_state(1)[0])
    result, refinement = legacy_search(graph, *args, refine_seed, remaining_steps,
                                       remaining_time, tolerance, search_null_num, null_estimator, verbose)
    for item in refinement:
        item = dict(item)
        item['phase'] = 'refinement'
        item['step'] = spent+item['step']
        history.append(item)
    return result, history


def search(initial, node_num, target, keys, max_r, max_v, seed, max_iter, max_seconds,
           tolerance, search_null_num, null_estimator, verbose=False, *, legacy_search):
    info = overlap_demand(node_num, len(initial), target, keys, max_v, tolerance)
    args = (node_num, target, keys, max_r, max_v)
    if not info['enriched'] or max_iter < 200:
        return legacy_search(initial, *args, seed, max_iter, max_seconds, tolerance,
                             search_null_num, null_estimator, verbose)
    start = time.perf_counter()
    graph, diagnostic = _compare_initializations(
        initial, *args, seed, max_iter, start+.2*max_seconds, null_estimator, search_null_num)
    if verbose:
        print('INITIALIZATION COMPARISON', diagnostic, flush=True)
    remaining = max_seconds-(time.perf_counter()-start)
    if remaining <= 0:
        return graph, [diagnostic]
    if diagnostic['switch']:
        result, history = _explore_and_refine(graph, *args, seed, max_iter-diagnostic['used'], remaining,
                                              tolerance, search_null_num, null_estimator, verbose, legacy_search=legacy_search)
    else:
        result, history = legacy_search(graph, *args, seed, max_iter-diagnostic['used'], remaining,
                                        tolerance, search_null_num, null_estimator, verbose)
    shifted = [diagnostic]
    for item in history:
        item = dict(item)
        for name in ('used', 'step'):
            if name in item:
                item[name] += diagnostic['used']
        shifted.append(item)
    return result, shifted
