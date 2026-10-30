"""Dense N/M/Delta reconstruction with independent configuration nulls.

No reference graph, degree sequence or edge-size sequence is an input.
Motif identities and repeated labelled hyperedges follow delta_reconstruction.
"""
import time
import numpy as np
import numba as nb
import delta_reconstruction as k



@nb.njit(cache=True)
def incidence(table, sizes, n):
    degree=np.zeros(n,np.int64)
    for i in range(len(sizes)):
        for p in range(sizes[i]):degree[table[i,p]]+=1
    ptr=np.zeros(n+1,np.int64)
    for v in range(n):ptr[v+1]=ptr[v]+degree[v]
    ids=np.empty(ptr[-1],np.int64);position=ptr[:-1].copy()
    for i in range(len(sizes)):
        for p in range(sizes[i]):
            v=table[i,p];ids[position[v]]=i;position[v]+=1
    return ptr,ids,degree


@nb.njit(cache=True)
def pair_and_degree(table,sizes,ptr,ids,pair,dimension):
    m=len(sizes)
    counts=np.zeros(dimension,np.float64)
    neighbor_degree=np.zeros(m,np.int64)
    mark=np.full(m,-1,np.int64);overlap=np.zeros(m,np.int64);touched=np.empty(m,np.int64)
    for i in range(m):
        used=0
        for pos in range(sizes[i]):
            v=table[i,pos]
            for p in range(ptr[v],ptr[v+1]):
                j=ids[p]
                if j==i:continue
                if mark[j]!=i:
                    mark[j]=i;overlap[j]=0;touched[used]=j;used+=1
                overlap[j]+=1
        neighbor_degree[i]=used
        for p in range(used):
            j=touched[p]
            if j>i:
                shared=overlap[j]
                index=pair[sizes[i]-shared+7*(sizes[j]-shared)+49*shared]
                counts[index]+=1.
    return counts,neighbor_degree


@nb.njit(cache=True)
def triple_id(table,sizes,i,j,h,triple):
    ab=k._overlap(table[i],sizes[i],table[j],sizes[j])
    ac=k._overlap(table[i],sizes[i],table[h],sizes[h])
    bc=k._overlap(table[j],sizes[j],table[h],sizes[h])
    shared=0
    if bc>0:
        for a in range(sizes[i]):
            v=table[i,a];in_b=False;in_c=False
            for b in range(sizes[j]):
                if table[j,b]==v:in_b=True;break
            if in_b:
                for c in range(sizes[h]):
                    if table[h,c]==v:in_c=True;break
            if in_b and in_c:shared+=1
    code=(sizes[i]-ab-ac+shared+7*(sizes[j]-ab-bc+shared)+49*(ab-shared)
          +343*(sizes[h]-ac-bc+shared)+2401*(ac-shared)+16807*(bc-shared)+117649*shared)
    return triple[code],(3 if bc else 1)


@nb.njit(cache=True)
def prepare(table,sizes,ptr,ids,pair,dimension):
    m=len(sizes);out=np.zeros(dimension,np.float64)
    categories=np.zeros((m,30),np.int32)
    marks=np.full(m,-1,np.int64);ov=np.zeros(m,np.int32);touch=np.empty(m,np.int32)
    for i in range(m):
        used=0
        for pos in range(sizes[i]):
            v=table[i,pos]
            for p in range(ptr[v],ptr[v+1]):
                j=ids[p]
                if j==i:continue
                if marks[j]!=i:
                    marks[j]=i;ov[j]=0;touch[used]=j;used+=1
                ov[j]+=1
        for p in range(used):
            j=touch[p];shared=ov[j]
            categories[i,(sizes[j]-2)*6+shared-1]+=1
            if j>i:out[pair[sizes[i]-shared+7*(sizes[j]-shared)+49*shared]]+=1
    offsets=np.zeros((m,31),np.int64)
    total=0
    for i in range(m):
        offsets[i,0]=total
        for a in range(30):
            total+=categories[i,a];offsets[i,a+1]=total
    if total>160000000:raise MemoryError('Projected adjacency exceeds the 160-million-entry memory bound.')
    neighbors=np.empty(total,np.int32);position=offsets[:,:30].copy()
    marks[:]=-1
    for i in range(m):
        used=0
        for pos in range(sizes[i]):
            v=table[i,pos]
            for p in range(ptr[v],ptr[v+1]):
                j=ids[p]
                if j==i:continue
                if marks[j]!=i:
                    marks[j]=i;ov[j]=0;touch[used]=j;used+=1
                ov[j]+=1
        for p in range(used):
            j=touch[p];cat=(sizes[j]-2)*6+ov[j]-1
            neighbors[position[i,cat]]=j;position[i,cat]+=1
    entries=np.zeros(4500,np.int64)
    for i in range(m):
        for a in range(30):
            if categories[i,a]==0:continue
            for b in range(a,30):
                count=categories[i,a]*(categories[i,b]-(a==b))
                if count>0:entries[(sizes[i]-2)*900+a*30+b]+=1
    group_ptr=np.zeros(4501,np.int64)
    for group in range(4500):group_ptr[group+1]=group_ptr[group]+entries[group]
    centers=np.empty(group_ptr[-1],np.int32);weights=np.empty(group_ptr[-1],np.float64)
    pos=group_ptr[:-1].copy()
    for i in range(m):
        for a in range(30):
            if categories[i,a]==0:continue
            for b in range(a,30):
                count=categories[i,a]*(categories[i,b]-(a==b))
                if count<=0:continue
                group=(sizes[i]-2)*900+a*30+b;p=pos[group]
                centers[p]=i;weights[p]=count/(2. if a==b else 1.);pos[group]+=1
    mass=np.zeros(4500,np.float64)
    for group in range(4500):
        total=0.
        for p in range(group_ptr[group],group_ptr[group+1]):
            total+=weights[p];weights[p]=total
        mass[group]=total
    return out,offsets,neighbors,group_ptr,centers,weights,mass


@nb.njit(cache=True)
def sample(table,sizes,offsets,neighbors,group_ptr,centers,cdf,allocation,triple,dimension,seed):
    np.random.seed(seed)
    result=np.zeros(dimension,np.float64);variance=np.zeros(dimension,np.float64)
    for group in range(4500):
        number=allocation[group]
        if not number:continue
        begin=group_ptr[group];end=group_ptr[group+1];mass=cdf[end-1]
        a=(group%900)//30;b=group%30
        sums=np.zeros(dimension,np.float64);squares=np.zeros(dimension,np.float64)
        for _ in range(number):
            value=np.random.random()*mass;lo=begin;hi=end
            while lo<hi:
                mid=(lo+hi)//2
                if cdf[mid]<=value:lo=mid+1
                else:hi=mid
            i=centers[min(lo,end-1)]
            left=np.random.randint(offsets[i,a],offsets[i,a+1])
            if a==b:
                right=np.random.randint(offsets[i,b],offsets[i,b+1]-1)
                if right>=left:right+=1
            else:right=np.random.randint(offsets[i,b],offsets[i,b+1])
            j=neighbors[left];h=neighbors[right]
            motif,witness=triple_id(table,sizes,i,j,h,triple)
            weight=1./witness;sums[motif]+=weight;squares[motif]+=weight*weight
        result+=mass*sums/number
        if number>1:
            variance+=mass*mass*np.maximum(0.,squares-sums*sums/number)/(number*(number-1))
    return result,variance


@nb.njit(cache=True)
def enumerate_groups(table,sizes,offsets,neighbors,gp,centers,chosen,triple,dimension):
    result=np.zeros(dimension,np.float64)
    for group in range(4500):
        if not chosen[group]:continue
        a=(group%900)//30;b=group%30
        for entry in range(gp[group],gp[group+1]):
            i=centers[entry]
            for left in range(offsets[i,a],offsets[i,a+1]):
                begin=left+1 if a==b else offsets[i,b]
                for right in range(begin,offsets[i,b+1]):
                    j=neighbors[left];h=neighbors[right]
                    motif,witness=triple_id(table,sizes,i,j,h,triple)
                    result[motif]+=1./witness
    return result


def counts(edges,n,keys,max_r=3,samples=200000,seed=1,return_se=False,factor=4.):
    table,sizes=k.pack(edges);pair,triple=k.lookup_tables(tuple(keys))
    ptr,ids,degree=incidence(table,sizes,n)
    if max_r == 2:
        result, _ = pair_and_degree(table, sizes, ptr, ids, pair, len(keys))
        return (result, np.zeros(len(keys))) if return_se else result
    result,offsets,neighbors,gp,centers,cdf,mass=prepare(table,sizes,ptr,ids,pair,len(keys))
    variance=np.zeros(len(keys));active=np.flatnonzero(mass>0)
    if max_r==3 and len(active):
        budget=max(int(samples),2*len(active))
        minimum=min(100,max(2,budget//len(active)))
        allocation=np.zeros(4500,np.int64);allocation[active]=minimum
        raw=(budget-int(allocation.sum()))*np.sqrt(mass[active])/np.sqrt(mass[active]).sum()
        allocation[active]+=np.floor(raw).astype(np.int64)
        order=active[np.argsort(-(raw-np.floor(raw)),kind='stable')]
        allocation[order[:budget-int(allocation.sum())]]+=1
        exact=(mass>0)&(mass<=factor*allocation)
        allocation[exact]=0
        result+=enumerate_groups(table,sizes,offsets,neighbors,gp,centers,exact,triple,len(keys))
        extra,variance=sample(table,sizes,offsets,neighbors,gp,centers,cdf,allocation,triple,len(keys),int(seed)%2**32)
        result+=extra
    return (result,np.sqrt(variance)) if return_se else result


@nb.njit(cache=True)
def edge_curveball(table,sizes,node_num,trades,seed):
    np.random.seed(seed)
    result=table.copy();m=len(sizes)
    if m<2:return result
    pool=np.empty(12,np.int64);common=np.empty(6,np.int64)
    for _ in range(trades):
        i=np.random.randint(m);j=np.random.randint(m-1)
        if j>=i:j+=1
        shared=0;used=0
        for a in range(sizes[i]):
            v=result[i,a];found=False
            for b in range(sizes[j]):
                if result[j,b]==v:found=True;break
            if found:common[shared]=v;shared+=1
            else:pool[used]=v;used+=1
        left=used
        for b in range(sizes[j]):
            v=result[j,b];found=False
            for a in range(sizes[i]):
                if result[i,a]==v:found=True;break
            if not found:pool[used]=v;used+=1
        if not used:continue
        for p in range(used-1,0,-1):
            q=np.random.randint(p+1);pool[p],pool[q]=pool[q],pool[p]
        for p in range(shared):result[i,p]=common[p];result[j,p]=common[p]
        for p in range(left):result[i,shared+p]=pool[p]
        for p in range(left,used):result[j,shared+p-left]=pool[p]
    return result


@nb.njit(cache=True)
def _block_seed(n,m,cap,probabilities,groups,locality,copying,heterogeneity,seed):
    np.random.seed(seed)
    table=np.full((m,6),-1,np.int64);sizes=np.full(m,2,np.int64)
    labels=np.random.permutation(n)
    groups=max(1,min(groups,n//cap))
    # Random vertex propensities; these are not read from the reference.
    propensity=np.random.gamma(heterogeneity,1.,n)+.01
    global_cdf=np.cumsum(propensity)
    cdf=np.cumsum(probabilities)
    for i in range(m):
        if i<n-1:
            table[i,0]=labels[i];table[i,1]=labels[i+1]
            continue
        s=2+np.searchsorted(cdf,np.random.random()*cdf[-1]);sizes[i]=s
        group=np.random.randint(groups)
        begin=group*n//groups;end=(group+1)*n//groups
        anchor=np.random.randint(i)
        for p in range(s):
            while True:
                if np.random.random()<copying:
                    vertex=table[anchor,np.random.randint(sizes[anchor])]
                elif np.random.random()<locality:
                    total=0.
                    for q in range(begin,end):total+=propensity[labels[q]]
                    chosen=np.random.random()*total;vertex=labels[end-1]
                    for q in range(begin,end):
                        vertex=labels[q];chosen-=propensity[vertex]
                        if chosen<=0:break
                else:
                    vertex=np.searchsorted(global_cdf,np.random.random()*global_cdf[-1])
                duplicate=False
                for q in range(p):
                    if table[i,q]==vertex:duplicate=True;break
                if not duplicate:break
            table[i,p]=vertex
    return table,sizes


def seed_graph(n,m,max_v,probabilities,groups,locality,copying,heterogeneity,seed):
    cap=min(max_v,n)
    p=np.asarray(probabilities[:cap-1],float);p=p/p.sum()
    if m<n-1:raise ValueError('This experimental dense constructor requires M >= N-1.')
    table,sizes=_block_seed(n,m,cap,p,int(groups),float(locality),float(copying),float(heterogeneity),int(seed)%2**32)
    return k.unpack(table,sizes)


def applicable(node_num, edge_num):
    """Keep the existing exact-search domain completely unchanged."""
    return edge_num > 1536 and edge_num >= node_num - 1 and node_num * edge_num <= 20000000


def distances(target, actual, keys):
    """Balance motif orders; also retain coordinate identity in the loss."""
    orders = np.asarray([key[0] for key in keys])
    w1 = np.asarray([np.mean(np.abs(np.sort(actual[orders == r]) -
                                   np.sort(target[orders == r])))
                     for r in np.unique(orders)])
    rmse = float(np.sqrt(np.dot(k.loss_weights(target, keys), (actual-target)**2)))
    return float(w1.mean() + .25 * rmse), w1, rmse


def evaluate(graph, node_num, target, keys, max_r, seed, nulls, samples,
             trades, deadline=None):
    """Every candidate uses its own margins and fresh null realizations."""
    def check_time():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Dense candidate scoring reached its time budget.')

    check_time()
    rng = np.random.default_rng(seed)
    observed = counts(graph, node_num, keys, max_r, samples, int(rng.integers(2**31)))
    table, sizes = k.pack(graph)
    degree = np.bincount(table[table >= 0], minlength=node_num)
    mean = np.zeros(len(keys))
    for _ in range(nulls):
        check_time()
        randomized = edge_curveball(table, sizes, node_num, trades, int(rng.integers(2**31)))
        if not np.array_equal(np.bincount(randomized[randomized >= 0], minlength=node_num), degree):
            raise AssertionError('Configuration null changed node degrees.')
        null = k.unpack(randomized, sizes)
        if any(len(set(edge)) != len(edge) for edge in null):
            raise AssertionError('Configuration null contains a repeated vertex.')
        check_time()
        mean += counts(null, node_num, keys, max_r, samples, int(rng.integers(2**31)))
    mean /= nulls
    delta = k.delta(observed, mean)
    objective, w1, rmse = distances(target, delta, keys)
    return dict(objective=objective, w1=w1, rmse=rmse, delta=delta,
                counts=observed, mean=mean)


def search(initial, node_num, target, keys, max_r, max_v, seed, max_iter,
           max_seconds, tolerance, search_null_num, sample_num, verbose=False):
    """Parameter exploration followed by independent, longer-null selection.

    Proposals change edge-size probabilities, grouping, locality, copying and
    degree heterogeneity. The original candidate is always retained. A candidate
    replaces it only if independent validation improves the combined objective
    and does not worsen either order's W1 or the weighted coordinate RMSE.
    This empirical guard is not a guarantee on unseen data or Monte Carlo error.
    The caller performs a separate final null computation after this returns.
    """
    initial = list(initial)
    started = time.perf_counter()
    deadline = started + max_seconds
    exploration_deadline = started + .65 * max_seconds
    trace = dict(proposals=0, verified=0, selected='initial', seconds=0.)
    if max_iter <= 0:
        return initial, trace
    m = len(initial)
    rng = np.random.default_rng(seed)
    graph_seed, scoring_seed = map(int, rng.integers(2**31, size=2))
    screening_nulls = min(3, search_null_num)
    screening_samples = max(300000, sample_num)
    archive = []
    try:
        for index in range(max_iter):
            if time.perf_counter() >= exploration_deadline:
                break
            if index < 16:
                slope = (0., .4, .8, 1.2)[index % 4]
                vector = np.r_[-slope*np.arange(5),
                               np.log((4., 8., 16., 32.)[index//4]),
                               2.2, (0., 1., 2., 3.)[index % 4], 0.]
            elif index < 32 or rng.random() < .2:
                p = rng.dirichlet(np.array([2., 1., .7, .5, .4]))
                vector = np.r_[np.log(np.maximum(p, 1e-5)),
                               rng.uniform(0., np.log(max(2, node_num//max_v))),
                               rng.uniform(0., 5.), rng.uniform(-3., 4.),
                               rng.uniform(-2., 2.5)]
            else:
                parent = archive[int(rng.integers(min(6, len(archive))))]['vector']
                fraction = max(index/max_iter, (time.perf_counter()-started)/(.65*max_seconds))
                scale = max(.25, 1.-fraction)
                vector = parent + rng.normal(0., [.8]*5+[.5, .8, .8, .5])*scale
            p = np.exp(vector[:5]-np.max(vector[:5]))
            p /= p.sum()
            groups = int(np.clip(round(np.exp(np.clip(vector[5], 0.,
                              np.log(max(2, node_num//max_v))))), 1, max(1, node_num//max_v)))
            locality = 1./(1.+np.exp(-np.clip(vector[6], -7., 7.)))
            copying = .98/(1.+np.exp(-np.clip(vector[7], -7., 7.)))
            heterogeneity = np.exp(np.clip(vector[8], -2.5, 3.))
            graph = seed_graph(node_num, m, max_v, p, groups, locality,
                               copying, heterogeneity, graph_seed)
            value = evaluate(graph, node_num, target, keys, max_r, scoring_seed,
                             screening_nulls, screening_samples, max(100, 100*m),
                             exploration_deadline)
            archive.append(dict(graph=graph, vector=vector, index=index,
                                objective=value['objective']))
            archive.sort(key=lambda row: row['objective'])
            del archive[8:]
            trace['proposals'] += 1
            if verbose and index % 10 == 0:
                print(f'  Dense search: {index+1} proposals, best screening objective '
                      f'{archive[0]["objective"]:.6f}', flush=True)
    except (TimeoutError, MemoryError):
        pass
    best = initial
    if archive:
        try:
            # Selection streams never reuse the screening streams. Final output
            # statistics are computed by the public caller on another stream.
            validation_seed = int(rng.integers(2**31))
            def verify(graph):
                return evaluate(graph, node_num, target, keys, max_r, validation_seed,
                                max(10, search_null_num), max(1000000, sample_num),
                                max(100, 1000*m), deadline)
            baseline = verify(initial)
            trace['baseline_objective'] = baseline['objective']
            best_value = baseline
            for row in archive[:4]:
                value = verify(row['graph'])
                trace['verified'] += 1
                if (value['objective'] < best_value['objective']
                        and np.all(value['w1'] <= baseline['w1'])
                        and value['rmse'] <= baseline['rmse']):
                    best, best_value = row['graph'], value
                    trace['selected'] = int(row['index'])
                if verbose:
                    print(f'  Dense independent selection: {trace["verified"]} candidates, '
                          f'objective {best_value["objective"]:.6f}', flush=True)
                if np.max(np.abs(value['delta']-target)) <= tolerance:
                    break
            trace['selected_objective'] = best_value['objective']
        except (TimeoutError, MemoryError):
            pass
    trace['seconds'] = time.perf_counter()-started
    return best, trace
