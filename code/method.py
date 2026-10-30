"""数据计算与实验方法。"""

from py import *
from dataprocess import *


# ------------------------方法目录------------------------
# 后续所有计算、实验和结果保存方法均写在本文件中。
# artifical_hypergraph_gen: 生成 random、small-world 和 preferential-attachment 人工超图


# ------------------------结果计算------------------------

# para 参数说明：
# 1. random hypergraph (net_type='r')
#    单个数值表示 edge_factor，即超边数 M 与节点数 N 的比例 M/N。
#    参数列表依次为 [edge_factor, cardinality_values, cardinality_probabilities,
#    seed, allow_duplicate_edges, name]。
# 2. small-world hypergraph (net_type='sw')
#    单个数值表示 rewiring_probability，即文献中的重连概率 p。
#    参数列表依次为 [degree, rewiring_probability, cardinality_values,
#    cardinality_probabilities, seed, allow_duplicate_edges, name]。
#    degree 对应环格中节点的邻接度 k，必须为正偶数。
# 3. preferential-attachment hypergraph (net_type='ba')
#    单个数值表示 vertex_arrival_probability，即文献中的节点到达概率 p。
#    参数列表依次为 [vertex_arrival_probability, cardinality_values,
#    cardinality_probabilities, initial_node_num, seed, allow_duplicate_edges,
#    allow_repeated_nodes, name]。
#    allow_repeated_nodes=True 时严格保留文献中“有放回选择已有节点”的设定；
#    False 时每条超边中的节点互异，更适合当前基于节点集合定义的 hyper-motif。
# 对需要明确指定参数名称的实验，建议直接传入字典；各模型支持的键见方法文档。
def artifical_hypergraph_gen(data_path: str, net_type: str, node_min = 10,
                             node_max = 10000, num = 100, para = None):
    '''
    生成人工超图，并将每个数据集保存到独立文件夹。

    Parameters
    ----------
    data_path : str
        数据集父目录。新数据集编号取现有文件夹最大编号加一，不填补编号空缺。
    net_type : str
        'r' 为 random hypergraph，'sw' 为 small-world hypergraph，
        'ba' 为 preferential-attachment/scale-free hypergraph。
    node_min, node_max : int
        每个网络节点数的闭区间。每次生成使用记录在 description.txt 中的随机种子，
        从该区间均匀抽取一个整数作为节点数。
    num : int
        生成网络的数量。
    para : number, list, tuple, dict or None
        None 使用默认参数；数值使用方法上方所列的单参数简写；list/tuple 使用方法
        上方所列的位置参数顺序。推荐使用 dict，完整键如下：

        r: edge_factor, edge_num, cardinality_values,
           cardinality_probabilities, seed, allow_duplicate_edges, name。
        sw: degree, rewiring_probability, cardinality_values,
            cardinality_probabilities, seed, allow_duplicate_edges, name。
        ba: vertex_arrival_probability, cardinality_values,
            cardinality_probabilities, initial_node_num, seed,
            allow_duplicate_edges, allow_repeated_nodes, name。

        cardinality_values 默认是 [2, 3, 4, 5, 6]，对应概率默认相等，因此三种
        模型均可生成包含不同超边大小的 non-uniform hypergraph。

    Returns
    -------
    list
        本次创建的数据集文件夹路径。

    Notes
    -----
    random 模型采用固定 M 的 non-uniform random hypergraph：先依据给定分布抽取
    每条超边的 cardinality，再从 N 个节点中等概率、无放回地选择相应节点。

    small-world 模型依据 Raghav, Boccaletti and Jalan (2023) 的环形规则超图与
    重连机制，并将原文的 Q-uniform 构造推广为逐边抽取 Q 的 mixed-cardinality
    构造。DOI: 10.1088/2632-072X/acf430。

    ba 模型依据 Avin et al. (2019) 的 H(H0, p, Y) 模型：每个时间步以概率 p
    新增节点和超边，否则只新增超边；已有节点按当前 hyperdegree 比例被选取，
    而 Y 决定新超边大小。DOI: 10.1145/3341161.3342867。

    link.csv 使用长表结构，每行是一条 (hyperedge_id, node_id) incidence；同一
    hyperedge_id 下重复出现同一 node_id 表示 PA 模型中的重复节点成员关系。
    '''
    net_type = str(net_type).lower()
    if net_type not in {'r', 'sw', 'ba'}:
        raise ValueError("net_type must be one of 'r', 'sw', or 'ba'.")
    if not isinstance(node_min, (int, np.integer)) or not isinstance(node_max, (int, np.integer)):
        raise TypeError('node_min and node_max must be integers.')
    if node_min < 2 or node_max < node_min:
        raise ValueError('node_min must be at least 2 and no larger than node_max.')
    if not isinstance(num, (int, np.integer)) or num < 1:
        raise ValueError('num must be a positive integer.')

    parameters = _func_artificial_normalize_parameters(net_type, para)
    os.makedirs(data_path, exist_ok = True)
    next_id = _func_artificial_next_dataset_id(data_path)
    result_path_list = []

    for result_index in range(num):
        random_seed = parameters['seed'] + result_index
        rng = np.random.default_rng(random_seed)
        node_num = int(rng.integers(node_min, node_max + 1))

        if max(parameters['cardinality_values']) > node_num:
            raise ValueError(
                f'cardinality cannot exceed node count: max cardinality '
                f'{max(parameters["cardinality_values"])} > {node_num}.'
            )

        if net_type == 'r':
            hyperedge_list = _func_artificial_random_hypergraph(node_num, parameters, rng)
        elif net_type == 'sw':
            hyperedge_list = _func_artificial_smallworld_hypergraph(node_num, parameters, rng)
        else:
            hyperedge_list = _func_artificial_pa_hypergraph(node_num, parameters, rng)

        dataset_id = next_id + result_index
        dataset_name = _func_artificial_dataset_name(
            dataset_id, net_type, node_num, parameters['name'], result_index
        )
        dataset_path = os.path.join(data_path, dataset_name)
        while os.path.exists(dataset_path):
            dataset_id += 1
            dataset_name = _func_artificial_dataset_name(
                dataset_id, net_type, node_num, parameters['name'], result_index
            )
            dataset_path = os.path.join(data_path, dataset_name)

        _func_artificial_save_dataset(
            dataset_path, dataset_id, dataset_name, net_type, node_num,
            hyperedge_list, parameters, random_seed
        )
        result_path_list.append(dataset_path)
        next_id = dataset_id - result_index

    return result_path_list


def _func_artificial_normalize_parameters(net_type: str, para):
    common = {
        'cardinality_values': [2, 3, 4, 5, 6],
        'cardinality_probabilities': None,
        'seed': 2026,
        'allow_duplicate_edges': False,
    }
    default_dict = {
        'r': {
            **common,
            'edge_factor': 1.0,
            'edge_num': None,
            'name': 'mixed_cardinality',
        },
        'sw': {
            **common,
            'degree': 4,
            'rewiring_probability': 0.1,
            'name': 'raghav',
        },
        'ba': {
            **common,
            'vertex_arrival_probability': 0.5,
            'initial_node_num': None,
            'allow_repeated_nodes': False,
            'name': 'avin',
        },
    }
    positional_key_dict = {
        'r': [
            'edge_factor', 'cardinality_values', 'cardinality_probabilities',
            'seed', 'allow_duplicate_edges', 'name'
        ],
        'sw': [
            'degree', 'rewiring_probability', 'cardinality_values',
            'cardinality_probabilities', 'seed', 'allow_duplicate_edges', 'name'
        ],
        'ba': [
            'vertex_arrival_probability', 'cardinality_values',
            'cardinality_probabilities', 'initial_node_num', 'seed',
            'allow_duplicate_edges', 'allow_repeated_nodes', 'name'
        ],
    }
    scalar_key_dict = {
        'r': 'edge_factor',
        'sw': 'rewiring_probability',
        'ba': 'vertex_arrival_probability',
    }

    parameters = default_dict[net_type].copy()
    if para is None:
        pass
    elif isinstance(para, dict):
        unknown_keys = set(para) - set(parameters)
        if unknown_keys:
            raise ValueError(f'Unsupported para keys for {net_type}: {sorted(unknown_keys)}')
        parameters.update(para)
    elif isinstance(para, (int, float, np.integer, np.floating)) and not isinstance(para, bool):
        parameters[scalar_key_dict[net_type]] = para
    elif isinstance(para, (list, tuple)):
        parameter_keys = positional_key_dict[net_type]
        if len(para) > len(parameter_keys):
            raise ValueError(
                f'Too many positional parameters for {net_type}; expected at most '
                f'{len(parameter_keys)}.'
            )
        parameters.update(dict(zip(parameter_keys, para)))
    else:
        raise TypeError('para must be a number, list, tuple, dict, or None.')

    cardinality_values = parameters['cardinality_values']
    if isinstance(cardinality_values, (int, np.integer)):
        cardinality_values = [int(cardinality_values)]
    else:
        cardinality_values = [int(value) for value in cardinality_values]
    if not cardinality_values or min(cardinality_values) < 2:
        raise ValueError('cardinality_values must contain integers no smaller than 2.')
    if len(set(cardinality_values)) != len(cardinality_values):
        raise ValueError('cardinality_values cannot contain duplicates.')

    probabilities = parameters['cardinality_probabilities']
    if probabilities is None:
        probabilities = [1 / len(cardinality_values)] * len(cardinality_values)
    else:
        probabilities = [float(value) for value in probabilities]
        if len(probabilities) != len(cardinality_values):
            raise ValueError(
                'cardinality_probabilities and cardinality_values must have the same length.'
            )
        if min(probabilities) < 0 or sum(probabilities) <= 0:
            raise ValueError('cardinality_probabilities must be nonnegative with a positive sum.')
        probability_sum = sum(probabilities)
        probabilities = [value / probability_sum for value in probabilities]

    order = np.argsort(cardinality_values)
    parameters['cardinality_values'] = [cardinality_values[index] for index in order]
    parameters['cardinality_probabilities'] = [probabilities[index] for index in order]
    parameters['seed'] = int(parameters['seed'])
    parameters['allow_duplicate_edges'] = bool(parameters['allow_duplicate_edges'])
    parameters['name'] = _func_artificial_safe_name(parameters['name'])

    if net_type == 'r':
        if parameters['edge_num'] is not None:
            parameters['edge_num'] = int(parameters['edge_num'])
            if parameters['edge_num'] < 1:
                raise ValueError('edge_num must be a positive integer.')
        parameters['edge_factor'] = float(parameters['edge_factor'])
        if parameters['edge_factor'] <= 0:
            raise ValueError('edge_factor must be positive.')
    elif net_type == 'sw':
        parameters['degree'] = int(parameters['degree'])
        parameters['rewiring_probability'] = float(parameters['rewiring_probability'])
        if parameters['degree'] < 2 or parameters['degree'] % 2 != 0:
            raise ValueError('small-world degree must be a positive even integer.')
        if not 0 <= parameters['rewiring_probability'] <= 1:
            raise ValueError('rewiring_probability must be between 0 and 1.')
    else:
        parameters['vertex_arrival_probability'] = float(
            parameters['vertex_arrival_probability']
        )
        if not 0 < parameters['vertex_arrival_probability'] <= 1:
            raise ValueError('vertex_arrival_probability must be in (0, 1].')
        if parameters['initial_node_num'] is not None:
            parameters['initial_node_num'] = int(parameters['initial_node_num'])
        parameters['allow_repeated_nodes'] = bool(parameters['allow_repeated_nodes'])

    return parameters


def _func_artificial_random_hypergraph(node_num: int, parameters: dict,
                                       rng: np.random.Generator):
    if parameters['edge_num'] is None:
        edge_num = max(1, int(round(node_num * parameters['edge_factor'])))
    else:
        edge_num = parameters['edge_num']

    if not parameters['allow_duplicate_edges']:
        edge_capacity = sum(
            math.comb(node_num, cardinality)
            for cardinality, probability in zip(
                parameters['cardinality_values'],
                parameters['cardinality_probabilities']
            )
            if probability > 0
        )
        if edge_num > edge_capacity:
            raise ValueError(
                f'edge_num={edge_num} exceeds the number of distinct allowed hyperedges '
                f'({edge_capacity}).'
            )

    hyperedge_list, hyperedge_set = [], set()
    attempt_num = 0
    max_attempt_num = max(10000, edge_num * 200)
    while len(hyperedge_list) < edge_num and attempt_num < max_attempt_num:
        attempt_num += 1
        cardinality = _func_artificial_sample_cardinality(parameters, rng)
        hyperedge = tuple(sorted(
            int(node) for node in rng.choice(node_num, size = cardinality, replace = False)
        ))
        if parameters['allow_duplicate_edges'] or hyperedge not in hyperedge_set:
            hyperedge_list.append(hyperedge)
            hyperedge_set.add(hyperedge)

    if len(hyperedge_list) != edge_num:
        raise RuntimeError(
            'Unable to sample the requested number of distinct random hyperedges. '
            'Reduce edge_num/edge_factor or allow duplicate hyperedges.'
        )
    return hyperedge_list


def _func_artificial_smallworld_hypergraph(node_num: int, parameters: dict,
                                           rng: np.random.Generator):
    degree = parameters['degree']
    max_cardinality = max(parameters['cardinality_values'])
    if degree // 2 + max_cardinality - 2 >= node_num:
        raise ValueError(
            'For the small-world ring lattice, degree / 2 + max cardinality - 2 '
            'must be smaller than the node count.'
        )

    original_entry_list, hyperedge_set = [], set()
    for anchor in range(node_num):
        for start_offset in range(1, degree // 2 + 1):
            cardinality = _func_artificial_sample_cardinality(parameters, rng)
            ordered_nodes = [anchor] + [
                (anchor + start_offset + offset) % node_num
                for offset in range(cardinality - 1)
            ]
            hyperedge = tuple(sorted(ordered_nodes))
            if parameters['allow_duplicate_edges'] or hyperedge not in hyperedge_set:
                original_entry_list.append((anchor, ordered_nodes))
                hyperedge_set.add(hyperedge)

    original_neighbor_dict = {node: set() for node in range(node_num)}
    for _, ordered_nodes in original_entry_list:
        for node in ordered_nodes:
            original_neighbor_dict[node].update(set(ordered_nodes) - {node})

    rewired_hyperedge_list = []
    occupied_hyperedges = set(hyperedge_set)
    rewiring_probability = parameters['rewiring_probability']
    all_nodes = set(range(node_num))

    for anchor, ordered_nodes in original_entry_list:
        old_hyperedge = tuple(sorted(ordered_nodes))
        if not parameters['allow_duplicate_edges']:
            occupied_hyperedges.discard(old_hyperedge)

        selected_hyperedge = old_hyperedge
        if rng.random() < rewiring_probability:
            candidate_nodes = list(
                all_nodes - original_neighbor_dict[anchor] - {anchor}
            )
            rng.shuffle(candidate_nodes)
            for new_node in candidate_nodes:
                rewired_nodes = ordered_nodes[:-1] + [int(new_node)]
                candidate_hyperedge = tuple(sorted(rewired_nodes))
                if len(set(candidate_hyperedge)) != len(candidate_hyperedge):
                    continue
                if (
                    parameters['allow_duplicate_edges']
                    or candidate_hyperedge not in occupied_hyperedges
                ):
                    selected_hyperedge = candidate_hyperedge
                    break

        rewired_hyperedge_list.append(selected_hyperedge)
        occupied_hyperedges.add(selected_hyperedge)

    return rewired_hyperedge_list


def _func_artificial_pa_hypergraph(node_num: int, parameters: dict,
                                   rng: np.random.Generator):
    max_cardinality = max(parameters['cardinality_values'])
    initial_node_num = parameters['initial_node_num']
    if initial_node_num is None:
        initial_node_num = max_cardinality
    if initial_node_num < max_cardinality:
        raise ValueError('initial_node_num must be at least max(cardinality_values).')
    if initial_node_num > node_num:
        raise ValueError('initial_node_num cannot exceed the generated node count.')

    hyperedge_list, hyperedge_set = [], set()
    node_degree = np.zeros(node_num, dtype = float)
    for anchor in range(initial_node_num):
        hyperedge = tuple(sorted(
            (anchor + offset) % initial_node_num
            for offset in range(max_cardinality)
        ))
        if hyperedge in hyperedge_set:
            continue
        hyperedge_list.append(hyperedge)
        hyperedge_set.add(hyperedge)
        for node in hyperedge:
            node_degree[node] += 1

    current_node_num = initial_node_num
    vertex_arrival_probability = parameters['vertex_arrival_probability']
    allow_duplicate_edges = parameters['allow_duplicate_edges']
    allow_repeated_nodes = parameters['allow_repeated_nodes']
    expected_step_num = (node_num - initial_node_num) / vertex_arrival_probability
    max_step_num = max(1000, int(math.ceil(expected_step_num * 100)))
    step_num = 0

    while current_node_num < node_num:
        step_num += 1
        if step_num > max_step_num:
            raise RuntimeError('PA generation exceeded its conservative maximum step count.')

        cardinality = _func_artificial_sample_cardinality(parameters, rng)
        add_new_node = rng.random() < vertex_arrival_probability
        selected_num = cardinality - 1 if add_new_node else cardinality
        probabilities = node_degree[:current_node_num]
        probabilities = probabilities / probabilities.sum()
        selected_nodes = rng.choice(
            current_node_num,
            size = selected_num,
            replace = allow_repeated_nodes,
            p = probabilities,
        ).tolist()

        if add_new_node:
            selected_nodes.append(current_node_num)
        hyperedge = tuple(sorted(int(node) for node in selected_nodes))

        if not allow_duplicate_edges and hyperedge in hyperedge_set:
            continue

        hyperedge_list.append(hyperedge)
        hyperedge_set.add(hyperedge)
        for node in hyperedge:
            node_degree[node] += 1
        if add_new_node:
            current_node_num += 1

    return hyperedge_list


def _func_artificial_sample_cardinality(parameters: dict,
                                        rng: np.random.Generator):
    return int(rng.choice(
        parameters['cardinality_values'],
        p = parameters['cardinality_probabilities'],
    ))


def _func_artificial_next_dataset_id(data_path: str):
    dataset_id_list = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if not os.path.isdir(file_path):
            continue
        prefix = file_name.split('.', 1)[0]
        if prefix.isdigit():
            dataset_id_list.append(int(prefix))
    return max(dataset_id_list, default = 0) + 1


def _func_artificial_dataset_name(dataset_id: int, net_type: str, node_num: int,
                                  base_name: str, result_index: int):
    type_name_dict = {
        'r': 'random',
        'sw': 'smallworld',
        'ba': 'scalefree',
    }
    network_name = f'{base_name}_{result_index + 1:03d}'
    return (
        f'{dataset_id}.a_{type_name_dict[net_type]}_{node_num}_{network_name}'
    )


def _func_artificial_safe_name(name):
    safe_name = ''.join(
        character if character.isalnum() or character in {'-', '_'} else '_'
        for character in str(name)
    ).strip('_')
    return safe_name or 'unnamed'


def _func_artificial_save_dataset(dataset_path: str, dataset_id: int,
                                  dataset_name: str, net_type: str, node_num: int,
                                  hyperedge_list: list, parameters: dict,
                                  random_seed: int):
    os.makedirs(dataset_path, exist_ok = False)
    link_path = os.path.join(dataset_path, 'link.csv')
    with open(link_path, 'w', newline = '', encoding = 'utf-8') as link_file:
        writer = csv.writer(link_file)
        writer.writerow(['hyperedge_id', 'node_id'])
        for hyperedge_id, hyperedge in enumerate(hyperedge_list):
            for node_id in hyperedge:
                writer.writerow([hyperedge_id, node_id])

    cardinality_count = Counter(len(hyperedge) for hyperedge in hyperedge_list)
    duplicate_edge_num = len(hyperedge_list) - len(set(hyperedge_list))
    repeated_node_edge_num = sum(
        len(hyperedge) != len(set(hyperedge)) for hyperedge in hyperedge_list
    )
    incidence_num = sum(len(hyperedge) for hyperedge in hyperedge_list)
    model_name_dict = {
        'r': 'Non-uniform random hypergraph (fixed M)',
        'sw': 'Mixed-cardinality Watts-Strogatz hypergraph',
        'ba': 'Random preferential-attachment hypergraph H(H0, p, Y)',
    }
    source_dict = {
        'r': (
            'Conditional on cardinality, every node subset is sampled uniformly; '
            'the cardinality is sampled from the configured distribution.'
        ),
        'sw': (
            'Raghav, Boccaletti and Jalan (2023), Smallworldness in hypergraphs, '
            'DOI: 10.1088/2632-072X/acf430; extended here to mixed cardinalities.'
        ),
        'ba': (
            'Avin, Lotker, Nahum and Peleg (2019), Random Preferential Attachment '
            'Hypergraph, DOI: 10.1145/3341161.3342867.'
        ),
    }

    description_path = os.path.join(dataset_path, 'description.txt')
    with open(description_path, 'w', encoding = 'utf-8') as description_file:
        description_file.write(f'Dataset ID: {dataset_id}\n')
        description_file.write(f'Dataset folder: {dataset_name}\n')
        description_file.write('Dataset class: artificial\n')
        description_file.write(f'Network type code: {net_type}\n')
        description_file.write(f'Network model: {model_name_dict[net_type]}\n')
        description_file.write(f'Node count: {node_num}\n')
        description_file.write(f'Hyperedge count: {len(hyperedge_list)}\n')
        description_file.write(f'Incidence count: {incidence_num}\n')
        description_file.write(
            f'Hyperedge cardinality counts: {dict(sorted(cardinality_count.items()))}\n'
        )
        description_file.write(f'Duplicate hyperedge count: {duplicate_edge_num}\n')
        description_file.write(
            f'Hyperedges containing repeated node memberships: '
            f'{repeated_node_edge_num}\n'
        )
        description_file.write(f'Random seed: {random_seed}\n')
        description_file.write(f'Model source: {source_dict[net_type]}\n')
        description_file.write('Generation parameters:\n')
        for key in sorted(parameters):
            if key != 'seed':
                description_file.write(f'  {key}: {parameters[key]}\n')
        description_file.write(
            'link.csv format: each row is one (hyperedge_id, node_id) incidence.\n'
        )


# ------------------------Hyper-motif analysis------------------------

_HM_SCHEMA_VERSION = '2026.08.27-hm-r23-v1'
_HM_DEFAULT_SEED = 2026
_HM_RESULT_FOLDER = {
    'null': '1.hypergraph_null_model_cal',
    'motif': '2.hypermotifs_sta_cal',
    'ci': '3.hypergraph_ci_cal',
    'deltai': '4.hypergraph_deltai_cal',
    'distribution_fig': '5.hypergraph_ci_deltai_dis_fig',
    'theory_fig': '6.hypergraph_ci_real_theory_fig',
    'compare_fig': '7.ar_hypergraph_compare_fig',
}


def _func_hm_dataset_sort_key(file_name):
    prefix = str(file_name).split('.', 1)[0]
    return (0, int(prefix)) if prefix.isdigit() else (1, str(file_name))


def _func_hm_dataset_list(data_path: str):
    result = []
    if not os.path.isdir(data_path):
        return result
    for file_name in sorted(os.listdir(data_path), key = _func_hm_dataset_sort_key):
        dataset_path = os.path.join(data_path, file_name)
        if (
            os.path.isdir(dataset_path)
            and os.path.isfile(os.path.join(dataset_path, 'link.csv'))
        ):
            result.append((file_name, dataset_path))
    return result


def _func_hm_parse_value(value):
    value = str(value).strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _func_hm_read_description(dataset_path: str, dataset_name = None):
    description_path = os.path.join(dataset_path, 'description.txt')
    metadata, parameters = {}, {}
    in_parameters = False
    if os.path.isfile(description_path):
        with open(description_path, 'r', encoding = 'utf-8-sig') as file:
            for raw_line in file:
                line = raw_line.rstrip('\r\n')
                if line.strip() == 'Generation parameters:':
                    in_parameters = True
                    continue
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key, value = key.strip(), _func_hm_parse_value(value)
                if in_parameters and raw_line[:1].isspace():
                    parameters[key] = value
                else:
                    in_parameters = False
                    metadata[key] = value

    dataset_name = dataset_name or os.path.basename(os.path.normpath(dataset_path))
    name_parts = dataset_name.split('_')
    class_token = name_parts[0].split('.', 1)[-1] if name_parts else ''
    dataset_class = str(metadata.get('Dataset class', '')).lower()
    if not dataset_class:
        dataset_class = 'artificial' if class_token == 'a' else 'real'

    network_code = str(metadata.get('Network type code', '')).lower()
    if not network_code and len(name_parts) > 1:
        type_token = name_parts[1].lower()
        network_code = {
            'random': 'r', 'smallworld': 'sw', 'scalefree': 'ba'
        }.get(type_token, type_token)

    result = {
        'dataset_name': dataset_name,
        'dataset_class': dataset_class,
        'network_type': network_code,
        'node_num': int(metadata.get('Node count', 0) or 0),
        'edge_num': int(metadata.get('Hyperedge count', 0) or 0),
        'random_seed': int(metadata.get('Random seed', _HM_DEFAULT_SEED) or _HM_DEFAULT_SEED),
        'parameters': parameters,
        'description': metadata,
    }
    return result


def _func_hm_read_hypergraph(dataset_path: str, dataset_name = None):
    metadata = _func_hm_read_description(dataset_path, dataset_name)
    edge_member_dict = defaultdict(list)
    link_path = os.path.join(dataset_path, 'link.csv')
    with open(link_path, 'r', newline = '', encoding = 'utf-8-sig') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {'hyperedge_id', 'node_id'} <= set(reader.fieldnames):
            raise ValueError(f'Invalid link.csv header: {link_path}')
        for row in reader:
            edge_member_dict[str(row['hyperedge_id'])].append(str(row['node_id']))

    def label_key(value):
        value_text = str(value)
        return (0, int(value_text)) if re.fullmatch(r'-?\d+', value_text) else (1, value_text)

    edge_id_list = sorted(edge_member_dict, key = label_key)
    raw_node_list = sorted(
        {node for members in edge_member_dict.values() for node in members},
        key = label_key,
    )
    numeric_nodes = all(re.fullmatch(r'\d+', node) for node in raw_node_list)
    numeric_value = [int(node) for node in raw_node_list] if numeric_nodes else []
    zero_based_contiguous = (
        numeric_nodes
        and numeric_value == list(range(len(numeric_value)))
    )
    if zero_based_contiguous:
        node_map = {node: int(node) for node in raw_node_list}
        inferred_node_num = max(node_map.values(), default = -1) + 1
    else:
        node_map = {node: index for index, node in enumerate(raw_node_list)}
        inferred_node_num = len(node_map)

    hyperedge_list = []
    for edge_id in edge_id_list:
        members = [node_map[node] for node in edge_member_dict[edge_id]]
        if len(members) != len(set(members)):
            raise ValueError(
                f'{metadata["dataset_name"]} contains a degenerate hyperedge: {edge_id}.'
            )
        hyperedge_list.append(tuple(sorted(members)))

    metadata['node_num'] = max(metadata['node_num'], inferred_node_num)
    metadata['edge_num'] = len(hyperedge_list)
    metadata['edge_id_list'] = edge_id_list
    return hyperedge_list, metadata


def _func_hm_source_fingerprint(dataset_path: str):
    digest = hashlib.sha256()
    for file_name in ['link.csv', 'description.txt']:
        file_path = os.path.join(dataset_path, file_name)
        digest.update(file_name.encode('utf-8'))
        if not os.path.isfile(file_path):
            continue
        with open(file_path, 'rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _func_hm_signature(settings):
    return hashlib.sha256(repr(settings).encode('utf-8')).hexdigest()


def _func_hm_atomic_pickle(data, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok = True)
    temp_path = f'{file_path}.tmp'
    with open(temp_path, 'wb') as file:
        pkl.dump(data, file, protocol = pkl.HIGHEST_PROTOCOL)
    os.replace(temp_path, file_path)


def _func_hm_manifest_path(folder_path: str):
    return os.path.join(folder_path, '_manifest.pkl')


def _func_hm_read_manifest(folder_path: str):
    manifest_path = _func_hm_manifest_path(folder_path)
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, 'rb') as file:
            manifest = pkl.load(file)
        return manifest if isinstance(manifest, dict) else {}
    except (OSError, EOFError, pkl.UnpicklingError):
        return {}


def _func_hm_is_complete(folder_path: str, dataset_name: str, file_path: str,
                         fingerprint: str, settings_signature: str):
    if not os.path.isfile(file_path):
        return False
    item = _func_hm_read_manifest(folder_path).get(dataset_name, {})
    return bool(
        item.get('complete')
        and item.get('schema_version') == _HM_SCHEMA_VERSION
        and item.get('source_fingerprint') == fingerprint
        and item.get('settings_signature') == settings_signature
    )


def _func_hm_mark_complete(folder_path: str, dataset_name: str,
                           fingerprint: str, settings_signature: str):
    manifest = _func_hm_read_manifest(folder_path)
    manifest[dataset_name] = {
        'complete': True,
        'schema_version': _HM_SCHEMA_VERSION,
        'source_fingerprint': fingerprint,
        'settings_signature': settings_signature,
        'completed_at': datetime.now().isoformat(timespec = 'seconds'),
    }
    _func_hm_atomic_pickle(manifest, _func_hm_manifest_path(folder_path))


def _func_hm_result_header(dataset_name: str, fingerprint: str, settings: dict,
                           metadata: dict, seed: int):
    return {
        'schema_version': _HM_SCHEMA_VERSION,
        'complete': True,
        'dataset_name': dataset_name,
        'source_fingerprint': fingerprint,
        'settings': settings,
        'settings_signature': _func_hm_signature(settings),
        'metadata': metadata,
        'seed': int(seed),
        'created_at': datetime.now().isoformat(timespec = 'seconds'),
    }


def _func_hm_seed(dataset_name: str, stage: str, seed = _HM_DEFAULT_SEED):
    digest = hashlib.sha256(f'{seed}|{dataset_name}|{stage}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder = 'little', signed = False)


def _func_hm_progress_text(start_time: float, item_index: int, item_total: int,
                           dataset_name: str):
    elapsed = max(time.time() - start_time, 1e-9)
    rate = item_index / elapsed if item_index else 0.0
    remaining = (item_total - item_index) / rate if rate else float('inf')
    finish = (
        datetime.now() + timedelta(seconds = remaining)
        if math.isfinite(remaining) else None
    )
    finish_text = finish.strftime('%Y-%m-%d %H:%M:%S') if finish else 'calculating'
    return f'network={dataset_name}, finish={finish_text}'


def _func_hm_atom_vector(hyperedge_tuple):
    edge_num = len(hyperedge_tuple)
    atom = [0] * (2 ** edge_num - 1)
    union_nodes = set().union(*(set(edge) for edge in hyperedge_tuple))
    edge_set_tuple = tuple(set(edge) for edge in hyperedge_tuple)
    for node in union_nodes:
        mask = 0
        for edge_index, edge in enumerate(edge_set_tuple):
            if node in edge:
                mask |= 1 << edge_index
        atom[mask - 1] += 1
    return tuple(atom)


def _func_hm_permute_atom(atom, permutation):
    edge_num = int(round(math.log2(len(atom) + 1)))
    transformed = [0] * len(atom)
    for old_mask in range(1, 2 ** edge_num):
        new_mask = 0
        for old_index in range(edge_num):
            if old_mask & (1 << old_index):
                new_mask |= 1 << permutation[old_index]
        transformed[new_mask - 1] = atom[old_mask - 1]
    return tuple(transformed)


def _func_hm_canonical_atom(atom):
    edge_num = int(round(math.log2(len(atom) + 1)))
    return min(
        _func_hm_permute_atom(atom, permutation)
        for permutation in permutations(range(edge_num))
    )


def _func_hm_automorphism_num(atom):
    edge_num = int(round(math.log2(len(atom) + 1)))
    return sum(
        _func_hm_permute_atom(atom, permutation) == tuple(atom)
        for permutation in permutations(range(edge_num))
    )


def _func_hm_atom_connected(atom):
    edge_num = int(round(math.log2(len(atom) + 1)))
    adjacency = [set() for _ in range(edge_num)]
    for mask, atom_size in enumerate(atom, start = 1):
        if atom_size <= 0:
            continue
        member_list = [i for i in range(edge_num) if mask & (1 << i)]
        for left, right in combinations(member_list, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    visited, stack = {0}, [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited) == edge_num


def _func_hm_atom_cardinalities(atom):
    edge_num = int(round(math.log2(len(atom) + 1)))
    return tuple(
        sum(
            atom[mask - 1]
            for mask in range(1, 2 ** edge_num)
            if mask & (1 << edge_index)
        )
        for edge_index in range(edge_num)
    )


def _func_hm_catalog(r_values = (2, 3), t = 6):
    catalog = {}
    if 2 in r_values:
        motif_set = set()
        for left_size in range(2, t + 1):
            for right_size in range(left_size, t + 1):
                for overlap_size in range(1, left_size + 1):
                    atom = (left_size - overlap_size,
                            right_size - overlap_size, overlap_size)
                    motif_set.add((2,) + _func_hm_canonical_atom(atom))
        catalog[2] = sorted(motif_set)

    if 3 in r_values:
        motif_set = set()
        for atom in product(range(t + 1), repeat = 7):
            cardinalities = _func_hm_atom_cardinalities(atom)
            if not all(2 <= value <= t for value in cardinalities):
                continue
            if not _func_hm_atom_connected(atom):
                continue
            motif_set.add((3,) + _func_hm_canonical_atom(atom))
        catalog[3] = sorted(motif_set)
    return catalog


def _func_hm_subset_count(hyperedge_list):
    subset_count = defaultdict(Counter)
    for hyperedge in hyperedge_list:
        edge_size = len(hyperedge)
        for subset_size in range(1, edge_size + 1):
            for subset in combinations(hyperedge, subset_size):
                subset_count[subset][edge_size] += 1
    return subset_count


def _func_hm_r2_exact(hyperedge_list, t = 6):
    eligible_edges = [edge for edge in hyperedge_list if 2 <= len(edge) <= t]
    subset_count = _func_hm_subset_count(eligible_edges)
    at_least = defaultdict(Counter)
    for subset, size_count in subset_count.items():
        subset_size = len(subset)
        size_list = sorted(size_count)
        for left_index, left_size in enumerate(size_list):
            for right_size in size_list[left_index:]:
                if left_size == right_size:
                    pair_num = math.comb(size_count[left_size], 2)
                else:
                    pair_num = size_count[left_size] * size_count[right_size]
                if pair_num:
                    at_least[subset_size][(left_size, right_size)] += pair_num

    result = Counter()
    for left_size in range(2, t + 1):
        for right_size in range(left_size, t + 1):
            exact_overlap = {}
            for overlap_size in range(min(left_size, right_size), 0, -1):
                value = at_least[overlap_size][(left_size, right_size)]
                for larger_overlap, larger_count in exact_overlap.items():
                    value -= math.comb(larger_overlap, overlap_size) * larger_count
                if value < 0 and abs(value) < 1e-9:
                    value = 0
                if value < 0:
                    raise ArithmeticError('Negative pair count in subset inversion.')
                exact_overlap[overlap_size] = int(value)
                if value:
                    atom = _func_hm_canonical_atom((
                        left_size - overlap_size,
                        right_size - overlap_size,
                        overlap_size,
                    ))
                    result[(2,) + atom] += int(value)
    return {
        'method': 'exact_subset_binomial_inversion',
        'counts': dict(result),
        'count_se': {key: 0.0 for key in result},
        'count_ci95': {key: (float(value), float(value)) for key, value in result.items()},
        'total': int(sum(result.values())),
        'sample_num': None,
        'effective_sample_size': None,
    }


def _func_hm_intersection_degree(hyperedge, subset_total):
    union_count = 0
    for subset_size in range(1, len(hyperedge) + 1):
        sign = 1 if subset_size % 2 else -1
        for subset in combinations(hyperedge, subset_size):
            union_count += sign * subset_total.get(subset, 0)
    return max(0, int(union_count - 1))


def _func_hm_uniform_neighbor(edge_index, hyperedge_list, node_to_edges,
                              node_degree, rng):
    edge = hyperedge_list[edge_index]
    witness_weight = np.array(
        [max(0, node_degree[node] - 1) for node in edge], dtype = float
    )
    total_weight = witness_weight.sum()
    if total_weight <= 0:
        return None
    witness_probability = witness_weight / total_weight
    for _ in range(1000):
        node = int(rng.choice(np.asarray(edge), p = witness_probability))
        candidate_list = node_to_edges[node]
        if len(candidate_list) <= 1:
            continue
        candidate = edge_index
        while candidate == edge_index:
            candidate = int(candidate_list[int(rng.integers(len(candidate_list)))])
        overlap_size = len(set(edge).intersection(hyperedge_list[candidate]))
        if overlap_size and rng.random() <= 1.0 / overlap_size:
            return candidate
    neighbor_set = set()
    for node in edge:
        neighbor_set.update(node_to_edges[node])
    neighbor_set.discard(edge_index)
    if not neighbor_set:
        return None
    neighbor_list = sorted(neighbor_set)
    return int(neighbor_list[int(rng.integers(len(neighbor_list)))])


def _func_hm_allocate_samples(stratum_weight, sample_num):
    active = {key: value for key, value in stratum_weight.items() if value > 0}
    if not active or sample_num <= 0:
        return {key: 0 for key in active}
    minimum = min(100, max(1, sample_num // len(active)))
    allocation = {key: minimum for key in active}
    remaining = max(0, sample_num - minimum * len(active))
    weight_sum = sum(active.values())
    raw = {key: remaining * value / weight_sum for key, value in active.items()}
    for key, value in raw.items():
        allocation[key] += int(math.floor(value))
    missing = sample_num - sum(allocation.values())
    order = sorted(active, key = lambda key: raw[key] - math.floor(raw[key]), reverse = True)
    for key in order[:missing]:
        allocation[key] += 1
    return allocation


def _func_hm_r3_sample(hyperedge_list, sample_num = 200000, seed = _HM_DEFAULT_SEED,
                       progress_desc = None):
    edge_num = len(hyperedge_list)
    node_to_edges = defaultdict(list)
    subset_total = Counter()
    for edge_index, hyperedge in enumerate(hyperedge_list):
        for node in hyperedge:
            node_to_edges[node].append(edge_index)
        for subset_size in range(1, len(hyperedge) + 1):
            for subset in combinations(hyperedge, subset_size):
                subset_total[subset] += 1
    node_degree = {node: len(edge_list) for node, edge_list in node_to_edges.items()}
    intersection_degree = np.zeros(edge_num, dtype = np.int64)
    stratum_center, stratum_weight = defaultdict(list), Counter()
    for edge_index, hyperedge in enumerate(hyperedge_list):
        degree = _func_hm_intersection_degree(hyperedge, subset_total)
        intersection_degree[edge_index] = degree
        if degree < 2:
            continue
        key = (len(hyperedge), int(math.floor(math.log2(degree))))
        wedge_num = math.comb(int(degree), 2)
        stratum_center[key].append((edge_index, wedge_num))
        stratum_weight[key] += wedge_num

    allocation = _func_hm_allocate_samples(stratum_weight, int(sample_num))
    rng = np.random.default_rng(seed)
    estimate, estimate_variance = Counter(), Counter()
    total_contribution, total_contribution_square = 0.0, 0.0
    importance_weight_list = []
    sample_progress = tqdm(
        total = sum(allocation.values()),
        desc = progress_desc or 'r=3 sampling',
        leave = False,
        dynamic_ncols = True,
        unit = 'sample',
    ) if progress_desc is not False else None

    for stratum_key in sorted(allocation):
        current_sample_num = allocation[stratum_key]
        if current_sample_num <= 0:
            continue
        center_data = stratum_center[stratum_key]
        center_index = np.asarray([item[0] for item in center_data], dtype = np.int64)
        center_probability = np.asarray([item[1] for item in center_data], dtype = float)
        center_probability /= center_probability.sum()
        wedge_total = float(stratum_weight[stratum_key])
        sum_value, sum_square = Counter(), Counter()
        total_value_sum, total_value_square_sum = 0.0, 0.0

        for _ in range(current_sample_num):
            center = int(rng.choice(center_index, p = center_probability))
            left = _func_hm_uniform_neighbor(
                center, hyperedge_list, node_to_edges, node_degree, rng
            )
            right = None
            for _attempt in range(100):
                right = _func_hm_uniform_neighbor(
                    center, hyperedge_list, node_to_edges, node_degree, rng
                )
                if right is not None and right != left:
                    break
            if left is None or right is None or left == right:
                if sample_progress is not None:
                    sample_progress.update(1)
                continue

            edge_tuple = (
                hyperedge_list[center], hyperedge_list[left], hyperedge_list[right]
            )
            atom = _func_hm_canonical_atom(_func_hm_atom_vector(edge_tuple))
            motif_key = (3,) + atom
            witness_num = 3 if set(hyperedge_list[left]).intersection(
                hyperedge_list[right]
            ) else 1
            value = 1.0 / witness_num
            sum_value[motif_key] += value
            sum_square[motif_key] += value * value
            total_value_sum += value
            total_value_square_sum += value * value
            importance_weight_list.append(wedge_total * value / current_sample_num)
            if sample_progress is not None:
                sample_progress.update(1)

        for motif_key, value_sum in sum_value.items():
            mean_value = value_sum / current_sample_num
            estimate[motif_key] += wedge_total * mean_value
            if current_sample_num > 1:
                sample_variance = max(
                    0.0,
                    (sum_square[motif_key] - current_sample_num * mean_value ** 2)
                    / (current_sample_num - 1),
                )
                estimate_variance[motif_key] += (
                    wedge_total ** 2 * sample_variance / current_sample_num
                )
        total_mean = total_value_sum / current_sample_num
        total_contribution += wedge_total * total_mean
        if current_sample_num > 1:
            total_sample_variance = max(
                0.0,
                (total_value_square_sum - current_sample_num * total_mean ** 2)
                / (current_sample_num - 1),
            )
            total_contribution_square += (
                wedge_total ** 2 * total_sample_variance / current_sample_num
            )

    if sample_progress is not None:
        sample_progress.close()
    count_se = {key: math.sqrt(estimate_variance[key]) for key in estimate}
    count_ci95 = {
        key: (
            max(0.0, estimate[key] - 1.96 * count_se[key]),
            estimate[key] + 1.96 * count_se[key],
        )
        for key in estimate
    }
    weight_array = np.asarray(importance_weight_list, dtype = float)
    effective_sample_size = (
        float(weight_array.sum() ** 2 / np.square(weight_array).sum())
        if weight_array.size and np.square(weight_array).sum() > 0 else 0.0
    )
    return {
        'method': 'stratified_wedge_importance_sampling',
        'counts': dict(estimate),
        'count_se': count_se,
        'count_ci95': count_ci95,
        'total': float(total_contribution),
        'total_se': math.sqrt(max(0.0, total_contribution_square)),
        'sample_num': int(sum(allocation.values())),
        'effective_sample_size': effective_sample_size,
        'strata': {
            key: {
                'wedge_num': int(stratum_weight[key]),
                'sample_num': int(allocation[key]),
            }
            for key in allocation
        },
    }


def alg_hypermotifs_mining(hyperedge_list: list, r_values = (2, 3), t = 6,
                           r3_sample_num = 200000, seed = _HM_DEFAULT_SEED,
                           progress_desc = None):
    '''统计恰好包含 r 条超边的 connected、unlabeled hyper-motifs。'''
    r_values = tuple(sorted(set(int(value) for value in r_values)))
    if not set(r_values) <= {2, 3}:
        raise ValueError('The current implementation supports r=2 and r=3.')
    eligible_edges = [
        tuple(sorted(set(edge))) for edge in hyperedge_list if 2 <= len(set(edge)) <= t
    ]
    result = {}
    if 2 in r_values:
        result[2] = _func_hm_r2_exact(eligible_edges, t = t)
    if 3 in r_values:
        result[3] = _func_hm_r3_sample(
            eligible_edges,
            sample_num = r3_sample_num,
            seed = seed,
            progress_desc = progress_desc,
        )
    return result


def _func_hm_build_curveball_state(hyperedge_list, node_num):
    edge_nodes = [set(edge) for edge in hyperedge_list]
    node_edges = [set() for _ in range(node_num)]
    for edge_index, edge in enumerate(edge_nodes):
        for node in edge:
            if node < 0 or node >= node_num:
                raise ValueError(f'Node id {node} is outside [0, {node_num}).')
            node_edges[node].add(edge_index)
    return {'edge_nodes': edge_nodes, 'node_edges': node_edges}


def _func_hm_copy_curveball_state(state):
    return {
        'edge_nodes': [set(edge) for edge in state['edge_nodes']],
        'node_edges': [set(edge_set) for edge_set in state['node_edges']],
    }


def _func_hm_curveball_trade(state, rng):
    node_num = len(state['node_edges'])
    if node_num < 2:
        return False
    left, right = rng.choice(node_num, size = 2, replace = False)
    left, right = int(left), int(right)
    left_incidence = state['node_edges'][left]
    right_incidence = state['node_edges'][right]
    common = left_incidence.intersection(right_incidence)
    left_only = list(left_incidence - common)
    right_only = list(right_incidence - common)
    if not left_only and not right_only:
        return False

    pool = left_only + right_only
    rng.shuffle(pool)
    new_left_only = set(pool[:len(left_only)])
    new_right_only = set(pool[len(left_only):])

    for edge_index in left_only:
        state['edge_nodes'][edge_index].discard(left)
    for edge_index in right_only:
        state['edge_nodes'][edge_index].discard(right)
    for edge_index in new_left_only:
        state['edge_nodes'][edge_index].add(left)
    for edge_index in new_right_only:
        state['edge_nodes'][edge_index].add(right)

    state['node_edges'][left] = common | new_left_only
    state['node_edges'][right] = common | new_right_only
    return new_left_only != set(left_only)


def _func_hm_curveball_trades(state, trade_num, rng, progress_bar = None):
    changed = 0
    update_block = 1000
    for trade_index in range(int(trade_num)):
        changed += int(_func_hm_curveball_trade(state, rng))
        if progress_bar is not None and (trade_index + 1) % update_block == 0:
            progress_bar.update(update_block)
    remainder = int(trade_num) % update_block
    if progress_bar is not None and remainder:
        progress_bar.update(remainder)
    return changed


def _func_hm_curveball_perturbation(state, original_edge_list):
    difference = 0
    incidence_num = 0
    for current, original in zip(state['edge_nodes'], original_edge_list):
        original_set = set(original)
        difference += len(current.symmetric_difference(original_set))
        incidence_num += len(original_set)
    return difference / (2.0 * incidence_num) if incidence_num else 0.0


def _func_hm_validate_curveball(state, original_node_degree,
                                original_cardinality, incidence_num):
    node_degree = np.asarray(
        [len(edge_set) for edge_set in state['node_edges']], dtype = np.int64
    )
    cardinality = np.asarray(
        [len(edge) for edge in state['edge_nodes']], dtype = np.int64
    )
    no_degenerate = all(len(edge) == len(set(edge)) for edge in state['edge_nodes'])
    result = {
        'node_degree_exact': bool(np.array_equal(node_degree, original_node_degree)),
        'cardinality_exact': bool(np.array_equal(cardinality, original_cardinality)),
        'incidence_exact': int(cardinality.sum()) == int(incidence_num),
        'no_degenerate_hyperedge': bool(no_degenerate),
        'duplicate_hyperedge_num': len(state['edge_nodes']) - len({
            tuple(sorted(edge)) for edge in state['edge_nodes']
        }),
    }
    if not all(result[key] for key in [
        'node_degree_exact', 'cardinality_exact', 'incidence_exact',
        'no_degenerate_hyperedge'
    ]):
        raise AssertionError(f'Hypercurveball invariant validation failed: {result}')
    return result


def _func_hm_estimate_curveball_mixing(original_state, original_edge_list,
                                       edge_num, node_num, seed,
                                       progress_desc = None):
    checkpoint_interval = max(1000, int(node_num))
    max_trade_num = max(checkpoint_interval * 10, int(100 * edge_num))
    state = _func_hm_copy_curveball_state(original_state)
    rng = np.random.default_rng(seed)
    perturbation_history = []
    progress_bar = tqdm(
        total = max_trade_num,
        desc = progress_desc or 'mixing calibration',
        leave = False,
        dynamic_ncols = True,
        unit = 'trade',
    )
    completed_trade_num = 0
    mixing_trade_num = max_trade_num
    while completed_trade_num < max_trade_num:
        block = min(checkpoint_interval, max_trade_num - completed_trade_num)
        _func_hm_curveball_trades(state, block, rng, progress_bar)
        completed_trade_num += block
        perturbation = _func_hm_curveball_perturbation(state, original_edge_list)
        perturbation_history.append((completed_trade_num, perturbation))
        if len(perturbation_history) >= 10:
            recent = np.asarray([item[1] for item in perturbation_history[-5:]], dtype = float)
            center = float(np.median(recent))
            within_range = (
                np.ptp(recent) <= 0.02 * max(abs(center), 1e-12)
            )
            previous = np.asarray(
                [item[1] for item in perturbation_history[-10:-5]], dtype = float
            )
            level_stable = abs(center - float(np.median(previous))) <= (
                0.02 * max(abs(center), 1e-12)
            )
            if within_range and level_stable:
                mixing_trade_num = completed_trade_num
                break
    progress_bar.close()
    return {
        'mixing_trade_num': int(mixing_trade_num),
        'checkpoint_interval': int(checkpoint_interval),
        'maximum_trade_num': int(max_trade_num),
        'perturbation_history': perturbation_history,
        'criterion': 'five checkpoints and preceding five-checkpoint plateau within 2 percent',
    }


def _func_hm_compress_curveball_state(state, node_num):
    dtype = np.dtype('<u2') if node_num <= np.iinfo(np.uint16).max else np.dtype('<u4')
    flat_nodes = np.asarray(
        [node for edge in state['edge_nodes'] for node in sorted(edge)], dtype = dtype
    )
    return {
        'dtype': dtype.str,
        'node_num': int(node_num),
        'compressed_nodes': zlib.compress(flat_nodes.tobytes(), level = 6),
    }


def _func_hm_decompress_null_sample(sample_data, cardinality):
    dtype = np.dtype(sample_data['dtype'])
    flat_nodes = np.frombuffer(
        zlib.decompress(sample_data['compressed_nodes']), dtype = dtype
    )
    hyperedge_list, position = [], 0
    for edge_size in cardinality:
        next_position = position + int(edge_size)
        hyperedge_list.append(tuple(int(node) for node in flat_nodes[position:next_position]))
        position = next_position
    if position != flat_nodes.size:
        raise ValueError('Compressed null model has an invalid incidence length.')
    return hyperedge_list


def hypergraph_null_model_cal(data_path: str, result_path: str, null_num = 100,
                              seed = _HM_DEFAULT_SEED, chain_num = 4):
    '''为 data 中每个网络生成 100 个 H_m(d) Hypercurveball null models。'''
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['null'])
    os.makedirs(output_folder, exist_ok = True)
    dataset_list = _func_hm_dataset_list(data_path)
    settings = {
        'null_num': int(null_num),
        'space': 'H_m(d)',
        'allow_duplicate_hyperedges': True,
        'allow_degenerate_hyperedges': False,
        'mixing_rule': 'adaptive_2_percent_plateau_max_100M',
        'chain_num': int(chain_num),
        'seed': int(seed),
    }
    settings_signature = _func_hm_signature(settings)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Null models total', dynamic_ncols = True, unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(
            output_folder, f'{dataset_name}_null_model.pkl'
        )
        if _func_hm_is_complete(
            output_folder, dataset_name, output_path, fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue

        hyperedge_list, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
        node_num = metadata['node_num']
        original_state = _func_hm_build_curveball_state(hyperedge_list, node_num)
        original_node_degree = np.asarray(
            [len(edge_set) for edge_set in original_state['node_edges']], dtype = np.int64
        )
        cardinality = np.asarray([len(edge) for edge in hyperedge_list], dtype = np.int32)
        incidence_num = int(cardinality.sum())
        dataset_seed = _func_hm_seed(dataset_name, 'null', seed)
        mixing = _func_hm_estimate_curveball_mixing(
            original_state, hyperedge_list, len(hyperedge_list), node_num,
            dataset_seed, progress_desc = f'{dataset_name} mixing',
        )
        mixing_trade_num = int(mixing['mixing_trade_num'])
        sample_gap = max(mixing['checkpoint_interval'], mixing_trade_num // 2)

        checkpoint_path = os.path.join(
            output_folder, f'{dataset_name}_null_model.checkpoint.pkl'
        )
        sample_list = []
        if os.path.isfile(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as file:
                    checkpoint = pkl.load(file)
                if (
                    checkpoint.get('source_fingerprint') == fingerprint
                    and checkpoint.get('settings_signature') == settings_signature
                    and checkpoint.get('mixing', {}).get('mixing_trade_num') == mixing_trade_num
                ):
                    sample_list = checkpoint.get('samples', [])
            except (OSError, EOFError, pkl.UnpicklingError):
                sample_list = []

        remaining_num = max(0, int(null_num) - len(sample_list))
        sample_bar = tqdm(
            total = int(null_num), initial = len(sample_list),
            desc = f'{dataset_name} null samples', leave = False,
            dynamic_ncols = True, unit = 'sample',
        )
        active_chain_num = (
            min(max(1, int(chain_num)), remaining_num) if remaining_num else 0
        )
        estimated_trade_num = (
            active_chain_num * mixing_trade_num
            + max(0, remaining_num - active_chain_num) * sample_gap
        )
        trade_bar = tqdm(
            total = estimated_trade_num,
            desc = f'{dataset_name} chain trades', leave = False,
            dynamic_ncols = True, unit = 'trade',
        )
        chain_data = []
        for chain_index in range(active_chain_num):
            chain_state = _func_hm_copy_curveball_state(original_state)
            chain_rng = np.random.default_rng(
                _func_hm_seed(dataset_name, f'null-chain-{chain_index}', seed)
            )
            _func_hm_curveball_trades(
                chain_state, mixing_trade_num, chain_rng, trade_bar
            )
            chain_data.append({
                'state': chain_state,
                'rng': chain_rng,
                'sample_num': 0,
            })

        while len(sample_list) < int(null_num):
            for chain_index, chain_item in enumerate(chain_data):
                if len(sample_list) >= int(null_num):
                    break
                chain_state = chain_item['state']
                chain_rng = chain_item['rng']
                if chain_item['sample_num'] > 0:
                    _func_hm_curveball_trades(
                        chain_state, sample_gap, chain_rng, trade_bar
                    )
                validation = _func_hm_validate_curveball(
                    chain_state, original_node_degree, cardinality, incidence_num
                )
                perturbation = _func_hm_curveball_perturbation(
                    chain_state, hyperedge_list
                )
                compressed = _func_hm_compress_curveball_state(chain_state, node_num)
                compressed.update({
                    'sample_id': len(sample_list),
                    'chain_id': chain_index,
                    'perturbation': float(perturbation),
                    'validation': validation,
                })
                sample_list.append(compressed)
                chain_item['sample_num'] += 1
                sample_bar.update(1)
                _func_hm_atomic_pickle({
                    'schema_version': _HM_SCHEMA_VERSION,
                    'source_fingerprint': fingerprint,
                    'settings_signature': settings_signature,
                    'mixing': mixing,
                    'samples': sample_list,
                }, checkpoint_path)
        trade_bar.close()
        sample_bar.close()

        perturbation = np.asarray(
            [item['perturbation'] for item in sample_list], dtype = float
        )
        lag1 = (
            float(np.corrcoef(perturbation[:-1], perturbation[1:])[0, 1])
            if perturbation.size > 2 and np.std(perturbation) > 0 else 0.0
        )
        result = _func_hm_result_header(
            dataset_name, fingerprint, settings, metadata, dataset_seed
        )
        result.update({
            'configuration_space': 'H_m(d)',
            'cardinality': cardinality,
            'node_degree': original_node_degree,
            'incidence_num': incidence_num,
            'mixing': mixing,
            'sample_gap_trade_num': int(sample_gap),
            'perturbation_lag1_autocorrelation': lag1,
            'autocorrelation_warning': bool(abs(lag1) > 0.1),
            'null_models': sample_list,
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, dataset_name, fingerprint, settings_signature
        )
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def _func_hm_pa_seed_edge_num(metadata, t):
    parameters = metadata.get('parameters', {})
    if parameters.get('seed_edge_num') is not None:
        return max(0, int(parameters['seed_edge_num']))
    initial_node_num = parameters.get('initial_node_num')
    if initial_node_num is None:
        initial_node_num = max(parameters.get('cardinality_values', [t]))
    initial_node_num = int(initial_node_num)
    max_cardinality = int(max(parameters.get('cardinality_values', [t])))
    seed_edges = {
        tuple(sorted(
            (anchor + offset) % initial_node_num
            for offset in range(max_cardinality)
        ))
        for anchor in range(initial_node_num)
    }
    return len(seed_edges)


def hypergraph_hypermotifs_sta_cal(data_path: str, result_path: str,
                                   r_values = (2, 3), t = 6,
                                   r3_sample_num = 200000,
                                   seed = _HM_DEFAULT_SEED):
    '''统计所有原始网络的 hyper-motif 数量。'''
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['motif'])
    os.makedirs(output_folder, exist_ok = True)
    catalog_path = os.path.join(output_folder, 'hypermotif_catalog.pkl')
    if not os.path.isfile(catalog_path):
        catalog_settings = {
            'r_values': tuple(r_values), 'maximum_cardinality': int(t)
        }
        _func_hm_atomic_pickle({
            'schema_version': _HM_SCHEMA_VERSION,
            'complete': True,
            'settings': catalog_settings,
            'settings_signature': _func_hm_signature(catalog_settings),
            'source_fingerprint': 'shared-catalog',
            'seed': None,
            'r_values': tuple(r_values),
            'maximum_cardinality': int(t),
            'catalog': _func_hm_catalog(r_values, t),
            'created_at': datetime.now().isoformat(timespec = 'seconds'),
        }, catalog_path)

    settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t),
        'r2_method': 'exact_subset_binomial_inversion',
        'r3_method': 'stratified_wedge_importance_sampling',
        'r3_sample_num': int(r3_sample_num), 'seed': int(seed),
    }
    settings_signature = _func_hm_signature(settings)
    dataset_list = _func_hm_dataset_list(data_path)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Original motifs total', dynamic_ncols = True,
        unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(
            output_folder, f'{dataset_name}_hypermotifs.pkl'
        )
        if _func_hm_is_complete(
            output_folder, dataset_name, output_path, fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        hyperedge_list, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
        dataset_seed = _func_hm_seed(dataset_name, 'original-motifs', seed)
        by_r = alg_hypermotifs_mining(
            hyperedge_list, r_values = r_values, t = t,
            r3_sample_num = r3_sample_num, seed = dataset_seed,
            progress_desc = f'{dataset_name} r=3',
        )
        theory_population = None
        if metadata['network_type'] == 'ba':
            seed_edge_num = _func_hm_pa_seed_edge_num(metadata, t)
            theory_population = {
                'excluded_seed_edge_num': seed_edge_num,
                'by_r': alg_hypermotifs_mining(
                    hyperedge_list[seed_edge_num:], r_values = r_values, t = t,
                    r3_sample_num = r3_sample_num,
                    seed = _func_hm_seed(dataset_name, 'post-seed-motifs', seed),
                    progress_desc = f'{dataset_name} post-seed r=3',
                ),
            }
        result = _func_hm_result_header(
            dataset_name, fingerprint, settings, metadata, dataset_seed
        )
        result.update({
            'by_r': by_r,
            'theory_comparison_population': theory_population,
            'unlabeled': True,
            'connected_only': True,
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, dataset_name, fingerprint, settings_signature
        )
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def hypergraph_null_model_hypermotifs_sta_cal(
        data_path: str, result_path: str, r_values = (2, 3), t = 6,
        r3_sample_num = 50000, seed = _HM_DEFAULT_SEED):
    '''统计所有 null models 的 hyper-motif 数量。'''
    null_folder = os.path.join(result_path, _HM_RESULT_FOLDER['null'])
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['motif'])
    os.makedirs(output_folder, exist_ok = True)
    settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t),
        'r2_method': 'exact_subset_binomial_inversion',
        'r3_method': 'stratified_wedge_importance_sampling',
        'r3_sample_num_per_null': int(r3_sample_num), 'seed': int(seed),
    }
    settings_signature = _func_hm_signature(settings)
    dataset_list = _func_hm_dataset_list(data_path)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Null motifs total', dynamic_ncols = True,
        unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(
            output_folder, f'{dataset_name}_null_model_hypermotifs.pkl'
        )
        if _func_hm_is_complete(
            output_folder, f'{dataset_name}:null-motifs', output_path,
            fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        null_path = os.path.join(null_folder, f'{dataset_name}_null_model.pkl')
        if not os.path.isfile(null_path):
            raise FileNotFoundError(f'Missing null model result: {null_path}')
        with open(null_path, 'rb') as file:
            null_data = pkl.load(file)
        cardinality = null_data['cardinality']
        metadata = null_data['metadata']
        checkpoint_path = os.path.join(
            output_folder, f'{dataset_name}_null_model_hypermotifs.checkpoint.pkl'
        )
        null_result_list = []
        if os.path.isfile(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as file:
                    checkpoint = pkl.load(file)
                if (
                    checkpoint.get('source_fingerprint') == fingerprint
                    and checkpoint.get('settings_signature') == settings_signature
                ):
                    null_result_list = checkpoint.get('null_results', [])
            except (OSError, EOFError, pkl.UnpicklingError):
                null_result_list = []

        sample_bar = tqdm(
            total = len(null_data['null_models']), initial = len(null_result_list),
            desc = f'{dataset_name} null motifs', leave = False,
            dynamic_ncols = True, unit = 'sample',
        )
        for sample_index in range(len(null_result_list), len(null_data['null_models'])):
            hyperedge_list = _func_hm_decompress_null_sample(
                null_data['null_models'][sample_index], cardinality
            )
            sample_result = alg_hypermotifs_mining(
                hyperedge_list, r_values = r_values, t = t,
                r3_sample_num = r3_sample_num,
                seed = _func_hm_seed(
                    dataset_name, f'null-motifs-{sample_index}', seed
                ),
                progress_desc = False,
            )
            null_result_list.append({
                'sample_id': sample_index,
                'by_r': sample_result,
            })
            sample_bar.update(1)
            _func_hm_atomic_pickle({
                'schema_version': _HM_SCHEMA_VERSION,
                'source_fingerprint': fingerprint,
                'settings_signature': settings_signature,
                'null_results': null_result_list,
            }, checkpoint_path)
        sample_bar.close()
        result = _func_hm_result_header(
            dataset_name, fingerprint, settings, metadata,
            _func_hm_seed(dataset_name, 'null-motifs', seed),
        )
        result.update({
            'null_num': len(null_result_list),
            'null_results': null_result_list,
            'unlabeled': True,
            'connected_only': True,
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, f'{dataset_name}:null-motifs', fingerprint,
            settings_signature
        )
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def _func_hm_load_catalog(result_path: str, r_values = (2, 3), t = 6):
    catalog_path = os.path.join(
        result_path, _HM_RESULT_FOLDER['motif'], 'hypermotif_catalog.pkl'
    )
    if os.path.isfile(catalog_path):
        with open(catalog_path, 'rb') as file:
            data = pkl.load(file)
        if (
            tuple(data.get('r_values', ())) == tuple(r_values)
            and int(data.get('maximum_cardinality', -1)) == int(t)
        ):
            return data['catalog']
    catalog = _func_hm_catalog(r_values, t)
    catalog_settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t)
    }
    _func_hm_atomic_pickle({
        'schema_version': _HM_SCHEMA_VERSION,
        'complete': True,
        'settings': catalog_settings,
        'settings_signature': _func_hm_signature(catalog_settings),
        'source_fingerprint': 'shared-catalog',
        'seed': None,
        'r_values': tuple(r_values),
        'maximum_cardinality': int(t),
        'catalog': catalog,
        'created_at': datetime.now().isoformat(timespec = 'seconds'),
    }, catalog_path)
    return catalog


def _func_hm_normalize_motif_result(motif_result, catalog_key_list):
    counts = motif_result.get('counts', {})
    count_se = motif_result.get('count_se', {})
    total = float(motif_result.get('total', sum(counts.values())))
    ci, ci_se, ci_ci95 = {}, {}, {}
    for motif_key in catalog_key_list:
        value = float(counts.get(motif_key, 0.0))
        standard_error = float(count_se.get(motif_key, 0.0))
        probability = value / total if total > 0 else 0.0
        probability_se = standard_error / total if total > 0 else 0.0
        ci[motif_key] = float(np.clip(probability, 0.0, 1.0))
        ci_se[motif_key] = max(0.0, probability_se)
        ci_ci95[motif_key] = (
            max(0.0, probability - 1.96 * probability_se),
            min(1.0, probability + 1.96 * probability_se),
        )
    probability_sum = sum(ci.values())
    if probability_sum > 0 and not np.isclose(probability_sum, 1.0, atol = 1e-8):
        ci = {key: value / probability_sum for key, value in ci.items()}
    return {
        'method': motif_result.get('method'),
        'counts': {key: float(counts.get(key, 0.0)) for key in catalog_key_list},
        'count_se': {key: float(count_se.get(key, 0.0)) for key in catalog_key_list},
        'total_count': total,
        'ci': ci,
        'ci_se': ci_se,
        'ci_ci95': ci_ci95,
        'sample_num': motif_result.get('sample_num'),
        'effective_sample_size': motif_result.get('effective_sample_size'),
    }


def hypergraph_ci_cal(data_path: str, result_path: str, r_values = (2, 3),
                      t = 6, seed = _HM_DEFAULT_SEED):
    '''由 motif 数量分别计算每个 r 下的 C_i。'''
    motif_folder = os.path.join(result_path, _HM_RESULT_FOLDER['motif'])
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['ci'])
    os.makedirs(output_folder, exist_ok = True)
    catalog = _func_hm_load_catalog(result_path, r_values, t)
    settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t),
        'normalization': 'within_each_exact_r', 'seed': int(seed),
    }
    settings_signature = _func_hm_signature(settings)
    dataset_list = _func_hm_dataset_list(data_path)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'C_i total', dynamic_ncols = True, unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(output_folder, f'{dataset_name}_ci.pkl')
        manifest_key = f'{dataset_name}:ci'
        if _func_hm_is_complete(
            output_folder, manifest_key, output_path, fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        motif_path = os.path.join(
            motif_folder, f'{dataset_name}_hypermotifs.pkl'
        )
        if not os.path.isfile(motif_path):
            raise FileNotFoundError(f'Missing motif statistics: {motif_path}')
        with open(motif_path, 'rb') as file:
            motif_data = pkl.load(file)
        by_r = {
            r: _func_hm_normalize_motif_result(motif_data['by_r'][r], catalog[r])
            for r in r_values
        }
        theory_population_by_r = None
        comparison_population = motif_data.get('theory_comparison_population')
        if comparison_population:
            theory_population_by_r = {
                r: _func_hm_normalize_motif_result(
                    comparison_population['by_r'][r], catalog[r]
                )
                for r in r_values
            }
        result = _func_hm_result_header(
            dataset_name, fingerprint, settings, motif_data['metadata'],
            _func_hm_seed(dataset_name, 'ci', seed),
        )
        result.update({
            'by_r': by_r,
            'theory_comparison_by_r': theory_population_by_r,
            'theory_population_note': (
                'PA comparison excludes deterministic seed hyperedges.'
                if theory_population_by_r else None
            ),
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, manifest_key, fingerprint, settings_signature
        )
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def _func_hm_log_falling(value, count):
    value = int(value)
    count = int(count)
    if count < 0 or value < count:
        return -np.inf
    if count == 0:
        return 0.0
    # Equation (S1.28): log((value)_count) = sum_i log(value - i).
    # The direct form supports arbitrary-size Python integers and avoids the
    # cancellation in lgamma(value + 1) - lgamma(value - count + 1).
    return float(math.fsum(
        math.log(value - offset) for offset in range(count)
    ))


def _func_hm_log_comb(value, count):
    if count < 0 or value < count:
        return -np.inf
    return float(
        gammaln(value + 1) - gammaln(count + 1) - gammaln(value - count + 1)
    )


def _func_hm_atom_has_equal_edges(atom):
    edge_num = int(round(math.log2(len(atom) + 1)))
    for left, right in combinations(range(edge_num), 2):
        equal = True
        for mask, atom_size in enumerate(atom, start = 1):
            left_member = bool(mask & (1 << left))
            right_member = bool(mask & (1 << right))
            if left_member != right_member and atom_size > 0:
                equal = False
                break
        if equal:
            return True
    return False


def _func_hm_random_theory(node_num, hyperedge_list, catalog, r_values,
                           t, allow_duplicate_edges):
    eligible_edges = [edge for edge in hyperedge_list if 2 <= len(edge) <= t]
    inventory = Counter(len(edge) for edge in eligible_edges)
    edge_num = len(eligible_edges)
    result = {}
    for r in r_values:
        log_probability = {}
        for motif_key in catalog[r]:
            atom = tuple(motif_key[1:])
            cardinalities = _func_hm_atom_cardinalities(atom)
            cardinality_count = Counter(cardinalities)
            log_cardinality = -_func_hm_log_falling(edge_num, r)
            feasible = True
            for cardinality, count in cardinality_count.items():
                value = _func_hm_log_falling(inventory[cardinality], count)
                if not np.isfinite(value):
                    feasible = False
                    break
                log_cardinality += value
            if not feasible:
                log_probability[motif_key] = -np.inf
                continue
            if not allow_duplicate_edges and _func_hm_atom_has_equal_edges(atom):
                log_probability[motif_key] = -np.inf
                continue

            union_size = sum(atom)
            if union_size > node_num:
                log_probability[motif_key] = -np.inf
                continue
            log_node_numerator = _func_hm_log_falling(node_num, union_size)
            log_node_numerator -= sum(gammaln(value + 1) for value in atom)
            if allow_duplicate_edges:
                log_node_denominator = sum(
                    _func_hm_log_comb(node_num, cardinality)
                    for cardinality in cardinalities
                )
            else:
                log_node_denominator = 0.0
                for cardinality, count in cardinality_count.items():
                    subset_num = math.comb(node_num, cardinality)
                    log_node_denominator += _func_hm_log_falling(subset_num, count)
            automorphism_num = _func_hm_automorphism_num(atom)
            orbit_num = math.factorial(r) / automorphism_num
            log_probability[motif_key] = (
                log_cardinality + log_node_numerator - log_node_denominator
                + math.log(orbit_num)
            )

        finite_log = np.asarray(
            [value for value in log_probability.values() if np.isfinite(value)],
            dtype = float,
        )
        normalizer = float(logsumexp(finite_log)) if finite_log.size else -np.inf
        probability = {
            key: (math.exp(value - normalizer) if np.isfinite(value) else 0.0)
            for key, value in log_probability.items()
        }
        result[r] = {
            'method': 'exact_fixed_inventory_random_law',
            'exact': True,
            'probability': probability,
            'probability_se': {key: 0.0 for key in probability},
            'probability_ci95': {key: (value, value) for key, value in probability.items()},
            'connected_normalized': True,
            'cardinality_inventory': dict(sorted(inventory.items())),
            'allow_duplicate_edges': bool(allow_duplicate_edges),
        }
    return result


def _func_hm_mc_probability(counter, connected_num, catalog_key_list, method,
                            assumption):
    probability, probability_se, probability_ci95 = {}, {}, {}
    for motif_key in catalog_key_list:
        value = counter.get(motif_key, 0)
        p_value = value / connected_num if connected_num else 0.0
        standard_error = (
            math.sqrt(max(0.0, p_value * (1.0 - p_value) / connected_num))
            if connected_num else 0.0
        )
        probability[motif_key] = p_value
        probability_se[motif_key] = standard_error
        probability_ci95[motif_key] = (
            max(0.0, p_value - 1.96 * standard_error),
            min(1.0, p_value + 1.96 * standard_error),
        )
    return {
        'method': method,
        'exact': False,
        'probability': probability,
        'probability_se': probability_se,
        'probability_ci95': probability_ci95,
        'connected_normalized': True,
        'connected_sample_num': int(connected_num),
        'assumption': assumption,
    }


def _func_hm_empty_theory(catalog, r_values, method, assumption):
    result = {}
    for r in r_values:
        probability = {key: 0.0 for key in catalog[r]}
        result[r] = {
            'method': method,
            'exact': False,
            'probability': probability,
            'probability_se': {key: 0.0 for key in probability},
            'probability_ci95': {
                key: (0.0, 0.0) for key in probability
            },
            'connected_normalized': True,
            'connected_sample_num': 0,
            'assumption': assumption,
        }
    return result


def _func_hm_smallworld_lattice_entries(node_num, hyperedge_list, degree):
    if not hyperedge_list:
        return []
    half_degree = max(1, int(degree) // 2)
    position_num = max(1, int(node_num) * half_degree)
    edge_num = len(hyperedge_list)
    result = []
    for edge_index, observed_edge in enumerate(hyperedge_list):
        # For a complete lattice this is the original anchor/offset ordering.
        # Partial lattices use evenly spaced positions and therefore retain N and M.
        position = min(
            position_num - 1,
            int(math.floor(edge_index * position_num / edge_num)),
        )
        anchor = position // half_degree
        start_offset = position % half_degree + 1
        ordered_nodes = [anchor] + [
            (anchor + start_offset + offset) % node_num
            for offset in range(len(observed_edge) - 1)
        ]
        result.append((anchor, tuple(ordered_nodes)))
    return result


def _func_hm_random_connected_pair_probability(node_num, inventory):
    edge_num = int(sum(inventory.values()))
    pair_num = math.comb(edge_num, 2) if edge_num >= 2 else 0
    if pair_num == 0:
        return 0.0
    connected_expectation = 0.0
    size_list = sorted(inventory)
    for left_index, left_size in enumerate(size_list):
        for right_size in size_list[left_index:]:
            if left_size == right_size:
                current_pair_num = math.comb(inventory[left_size], 2)
            else:
                current_pair_num = (
                    inventory[left_size] * inventory[right_size]
                )
            if current_pair_num == 0:
                continue
            if node_num - left_size < right_size:
                disjoint_probability = 0.0
            else:
                disjoint_probability = math.exp(
                    _func_hm_log_comb(node_num - left_size, right_size)
                    - _func_hm_log_comb(node_num, right_size)
                )
            connected_expectation += current_pair_num * (
                1.0 - disjoint_probability
            )
    return float(np.clip(connected_expectation / pair_num, 0.0, 1.0))


def _func_hm_real_model_inputs(node_num, hyperedge_list, metadata, t,
                               observed_connected_pair_num = None):
    eligible_edges = [edge for edge in hyperedge_list if 2 <= len(edge) <= t]
    inventory = Counter(len(edge) for edge in eligible_edges)
    edge_num = len(eligible_edges)
    cardinality_values = sorted(inventory)
    cardinality_probabilities = (
        [inventory[value] / edge_num for value in cardinality_values]
        if edge_num else []
    )
    allow_duplicate_edges = len(set(eligible_edges)) < edge_num

    model_input = {
        'observed_node_num': int(node_num),
        'observed_eligible_edge_num': int(edge_num),
        'cardinality_inventory': dict(sorted(inventory.items())),
        'allow_duplicate_edges': allow_duplicate_edges,
    }
    if not eligible_edges:
        return {
            'eligible_edges': eligible_edges,
            'random_allow_duplicate_edges': allow_duplicate_edges,
            'smallworld_metadata': None,
            'pa_metadata': None,
            'pa_reference_edges': [],
            'model_input': model_input,
        }

    maximum_cardinality = max(cardinality_values)
    maximum_half_degree = max(1, node_num - maximum_cardinality + 1)
    half_degree = max(1, int(math.ceil(edge_num / max(node_num, 1))))
    half_degree = min(half_degree, maximum_half_degree)
    smallworld_degree = 2 * half_degree
    lattice_entries = _func_hm_smallworld_lattice_entries(
        node_num, eligible_edges, smallworld_degree
    )
    lattice_edges = [
        tuple(sorted(ordered_nodes))
        for _anchor, ordered_nodes in lattice_entries
    ]
    pair_num = math.comb(edge_num, 2) if edge_num >= 2 else 0
    if pair_num:
        if observed_connected_pair_num is None:
            observed_connected_pair_num = _func_hm_r2_exact(
                eligible_edges, t = t
            )['total']
        lattice_connected_pair_num = _func_hm_r2_exact(
            lattice_edges, t = t
        )['total']
        observed_pair_probability = observed_connected_pair_num / pair_num
        lattice_pair_probability = lattice_connected_pair_num / pair_num
        random_pair_probability = _func_hm_random_connected_pair_probability(
            node_num, inventory
        )
        denominator = lattice_pair_probability - random_pair_probability
        if abs(denominator) <= np.finfo(float).eps:
            rewiring_probability = (
                0.0
                if abs(observed_pair_probability - lattice_pair_probability)
                <= abs(observed_pair_probability - random_pair_probability)
                else 1.0
            )
        else:
            retained_local_pair_fraction = np.clip(
                (observed_pair_probability - random_pair_probability)
                / denominator,
                0.0, 1.0,
            )
            # Both rows of a local pair must avoid rewiring, hence (1-omega)^2.
            rewiring_probability = 1.0 - math.sqrt(
                float(retained_local_pair_fraction)
            )
    else:
        observed_pair_probability = 0.0
        lattice_pair_probability = 0.0
        random_pair_probability = 0.0
        rewiring_probability = 0.0

    smallworld_metadata = dict(metadata)
    smallworld_metadata['parameters'] = {
        'degree': int(smallworld_degree),
        'rewiring_probability': float(rewiring_probability),
        'cardinality_values': cardinality_values,
        'cardinality_probabilities': cardinality_probabilities,
        'condition_on_cardinality_inventory': True,
        'allow_duplicate_edges': allow_duplicate_edges,
    }

    seed_edge_num = 1
    post_seed_edge_num = max(0, edge_num - seed_edge_num)
    initial_node_num = max(maximum_cardinality, node_num - post_seed_edge_num)
    initial_node_num = min(node_num, initial_node_num)
    birth_num = node_num - initial_node_num
    if post_seed_edge_num:
        vertex_arrival_probability = max(
            np.finfo(float).eps,
            birth_num / post_seed_edge_num,
        )
    else:
        vertex_arrival_probability = 1.0

    post_seed_inventory = inventory.copy()
    post_seed_inventory[maximum_cardinality] -= 1
    if post_seed_inventory[maximum_cardinality] <= 0:
        del post_seed_inventory[maximum_cardinality]
    pa_cardinality_values = sorted(post_seed_inventory)
    pa_post_edge_num = sum(post_seed_inventory.values())
    pa_cardinality_probabilities = (
        [
            post_seed_inventory[value] / pa_post_edge_num
            for value in pa_cardinality_values
        ]
        if pa_post_edge_num else [1.0]
    )
    if not pa_cardinality_values:
        pa_cardinality_values = [maximum_cardinality]

    pa_metadata = dict(metadata)
    pa_metadata['parameters'] = {
        'vertex_arrival_probability': float(vertex_arrival_probability),
        'cardinality_values': pa_cardinality_values,
        'cardinality_probabilities': pa_cardinality_probabilities,
        'initial_node_num': int(initial_node_num),
        'seed_edge_num': int(seed_edge_num),
        'condition_on_cardinality_inventory': True,
        'allow_duplicate_edges': allow_duplicate_edges,
        'allow_repeated_nodes': False,
    }
    pa_reference_edges = [tuple(range(maximum_cardinality))]
    for cardinality, count in sorted(post_seed_inventory.items()):
        pa_reference_edges.extend(
            [tuple(range(cardinality))] * int(count)
        )

    model_input.update({
        'smallworld': {
            'degree': int(smallworld_degree),
            'rewiring_probability': float(rewiring_probability),
            'rewiring_calibration': 'connected_pair_fraction_interpolation',
            'observed_connected_pair_probability': float(
                observed_pair_probability
            ),
            'lattice_connected_pair_probability': float(
                lattice_pair_probability
            ),
            'random_connected_pair_probability': float(
                random_pair_probability
            ),
        },
        'ba': {
            'initial_node_num': int(initial_node_num),
            'post_seed_edge_num': int(post_seed_edge_num),
            'vertex_arrival_probability': float(vertex_arrival_probability),
            'birth_num': int(birth_num),
            'calibration': (
                'N, M and the post-seed observed cardinality inventory'
            ),
        },
    })
    return {
        'eligible_edges': eligible_edges,
        'random_allow_duplicate_edges': allow_duplicate_edges,
        'smallworld_metadata': smallworld_metadata,
        'pa_metadata': pa_metadata,
        'pa_reference_edges': pa_reference_edges,
        'model_input': model_input,
    }


def _func_hm_smallworld_theory(node_num, hyperedge_list, metadata, catalog,
                               r_values, theory_sample_num, seed):
    parameters = metadata.get('parameters', {})
    degree = int(parameters.get('degree', 4))
    rewiring_probability = float(parameters.get('rewiring_probability', 0.1))
    allow_duplicate_edges = bool(parameters.get('allow_duplicate_edges', False))
    condition_on_inventory = bool(
        parameters.get('condition_on_cardinality_inventory', False)
    )
    if not hyperedge_list:
        return _func_hm_empty_theory(
            catalog, r_values, 'empty_smallworld_model',
            'No eligible hyperedges are available for the small-world model.',
        )
    original_entries = _func_hm_smallworld_lattice_entries(
        node_num, hyperedge_list, degree
    )

    rng = np.random.default_rng(seed)
    replicate_num = min(20, max(4, int(math.ceil(theory_sample_num / 10000))))
    r3_sample_per_replicate = max(1000, int(math.ceil(
        theory_sample_num / replicate_num
    )))
    replicate_result = []
    progress_bar = tqdm(
        total = replicate_num,
        desc = f'{metadata["dataset_name"]} SW kernel integration',
        leave = False, dynamic_ncols = True, unit = 'replicate',
    )
    for replicate_index in range(replicate_num):
        if condition_on_inventory:
            cardinality_order = rng.permutation(len(hyperedge_list))
            replicate_edge_list = [
                hyperedge_list[int(index)] for index in cardinality_order
            ]
            current_entries = _func_hm_smallworld_lattice_entries(
                node_num, replicate_edge_list, degree
            )
        else:
            current_entries = original_entries
        original_neighbor = [set() for _ in range(node_num)]
        for _anchor, ordered_nodes in current_entries:
            edge_set = set(ordered_nodes)
            for node in edge_set:
                original_neighbor[node].update(edge_set - {node})
        occupied = {
            tuple(sorted(ordered_nodes)) for _anchor, ordered_nodes in current_entries
        }
        sampled_hyperedges = []
        for anchor, ordered_nodes in current_entries:
            old_hyperedge = tuple(sorted(ordered_nodes))
            if not allow_duplicate_edges:
                occupied.discard(old_hyperedge)
            selected_hyperedge = old_hyperedge
            if rng.random() < rewiring_probability:
                for _ in range(max(1000, node_num * 2)):
                    endpoint = int(rng.integers(node_num))
                    if endpoint == anchor or endpoint in original_neighbor[anchor]:
                        continue
                    candidate = tuple(sorted(ordered_nodes[:-1] + (endpoint,)))
                    if len(candidate) != len(set(candidate)):
                        continue
                    if allow_duplicate_edges or candidate not in occupied:
                        selected_hyperedge = candidate
                        break
            sampled_hyperedges.append(selected_hyperedge)
            occupied.add(selected_hyperedge)
        replicate_result.append(alg_hypermotifs_mining(
            sampled_hyperedges, r_values = r_values,
            t = max(len(edge) for edge in hyperedge_list),
            r3_sample_num = r3_sample_per_replicate,
            seed = _func_hm_seed(
                metadata['dataset_name'], f'sw-theory-{replicate_index}', seed
            ),
            progress_desc = False,
        ))
        progress_bar.update(1)
    progress_bar.close()

    result = {}
    for r in r_values:
        key_list = catalog[r]
        count_matrix = np.asarray([
            [item[r]['counts'].get(key, 0.0) for key in key_list]
            for item in replicate_result
        ], dtype = float)
        total_by_replicate = count_matrix.sum(axis = 1)
        aggregate_count = count_matrix.sum(axis = 0)
        aggregate_total = aggregate_count.sum()
        probability = np.divide(
            aggregate_count, aggregate_total,
            out = np.zeros_like(aggregate_count), where = aggregate_total > 0,
        )
        probability_matrix = np.divide(
            count_matrix,
            total_by_replicate[:, None],
            out = np.zeros_like(count_matrix),
            where = total_by_replicate[:, None] > 0,
        )
        probability_se = (
            np.std(probability_matrix, axis = 0, ddof = 1) / math.sqrt(replicate_num)
            if replicate_num > 1 else np.zeros_like(probability)
        )
        if r == 3:
            sampling_variance = np.asarray([
                [
                    item[r].get('count_se', {}).get(key, 0.0) ** 2
                    for key in key_list
                ]
                for item in replicate_result
            ], dtype = float)
            probability_se = np.sqrt(
                np.square(probability_se)
                + np.sum(sampling_variance, axis = 0)
                / max(aggregate_total, 1.0) ** 2
            )
        result[r] = {
            'method': 'mixed_cardinality_smallworld_kernel_replicate_integration',
            'exact': False,
            'probability': dict(zip(key_list, probability.tolist())),
            'probability_se': dict(zip(key_list, probability_se.tolist())),
            'probability_ci95': {
                key: (
                    max(0.0, float(probability[index] - 1.96 * probability_se[index])),
                    min(1.0, float(probability[index] + 1.96 * probability_se[index])),
                )
                for index, key in enumerate(key_list)
            },
            'connected_normalized': True,
            'replicate_num': replicate_num,
            'r3_sample_per_replicate': (
                r3_sample_per_replicate if r == 3 else None
            ),
            'rewiring_probability': rewiring_probability,
            'condition_on_cardinality_inventory': condition_on_inventory,
            'assumption': (
                'Conditions on observed identity cardinalities and numerically integrates '
                'the sequential mixed-cardinality rewiring kernel. Duplicate suppression '
                'matches the dataset generation setting.'
            ),
        }
    return result


def _func_hm_fenwick_add(fenwick_tree, node_index, difference):
    tree_index = int(node_index) + 1
    while tree_index < len(fenwick_tree):
        fenwick_tree[tree_index] += difference
        tree_index += tree_index & -tree_index


def _func_hm_fenwick_total(fenwick_tree):
    total = 0.0
    tree_index = len(fenwick_tree) - 1
    while tree_index > 0:
        total += fenwick_tree[tree_index]
        tree_index -= tree_index & -tree_index
    return float(total)


def _func_hm_fenwick_find(fenwick_tree, target):
    tree_index = 0
    step = 1 << ((len(fenwick_tree) - 1).bit_length() - 1)
    while step:
        next_index = tree_index + step
        if next_index < len(fenwick_tree) and fenwick_tree[next_index] <= target:
            target -= fenwick_tree[next_index]
            tree_index = next_index
        step >>= 1
    return min(tree_index, len(fenwick_tree) - 2)


def _func_hm_weighted_without_replacement(fenwick_tree, node_weight,
                                          active_node_num, sample_num, rng):
    if sample_num <= 0:
        return []
    if active_node_num < sample_num:
        return None

    selected, tree_backup = [], {}
    for _ in range(sample_num):
        total_weight = _func_hm_fenwick_total(fenwick_tree)
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            for tree_index, value in tree_backup.items():
                fenwick_tree[tree_index] = value
            return None
        draw = min(
            float(rng.random() * total_weight),
            float(np.nextafter(total_weight, 0.0)),
        )
        node_index = _func_hm_fenwick_find(fenwick_tree, draw)
        weight = float(node_weight[node_index])
        if weight <= 0.0 or not np.isfinite(weight):
            for tree_index, value in tree_backup.items():
                fenwick_tree[tree_index] = value
            return None
        selected.append(node_index)
        tree_index = node_index + 1
        while tree_index < len(fenwick_tree):
            if tree_index not in tree_backup:
                tree_backup[tree_index] = fenwick_tree[tree_index]
            fenwick_tree[tree_index] -= weight
            tree_index += tree_index & -tree_index

    for tree_index, value in tree_backup.items():
        fenwick_tree[tree_index] = value
    return selected


def _func_hm_pa_meanfield_theory(node_num, hyperedge_list, metadata, catalog,
                                 r_values, theory_sample_num, t, seed):
    parameters = metadata.get('parameters', {})
    eta = float(parameters.get('vertex_arrival_probability', 0.5))
    cardinality_value = np.asarray(
        parameters.get('cardinality_values', list(range(2, t + 1))), dtype = int
    )
    cardinality_probability = np.asarray(
        parameters.get(
            'cardinality_probabilities',
            [1 / len(cardinality_value)] * len(cardinality_value),
        ), dtype = float,
    )
    cardinality_probability /= cardinality_probability.sum()
    initial_node_num = parameters.get('initial_node_num')
    if initial_node_num is None:
        initial_node_num = int(max(cardinality_value))
    initial_node_num = int(initial_node_num)
    seed_edge_num = _func_hm_pa_seed_edge_num(metadata, t)
    post_seed_edge_num = max(1, len(hyperedge_list) - seed_edge_num)
    seed_edges = hyperedge_list[:seed_edge_num]
    condition_on_inventory = bool(
        parameters.get('condition_on_cardinality_inventory', False)
    )
    post_seed_cardinality = np.asarray([
        len(edge) for edge in hyperedge_list[seed_edge_num:]
    ], dtype = int)
    seed_degree = np.zeros(initial_node_num, dtype = float)
    for edge in seed_edges:
        for node in edge:
            if node < initial_node_num:
                seed_degree[node] += 1
    seed_degree[seed_degree <= 0] = 1.0
    total_seed_degree = float(seed_degree.sum())
    mean_cardinality = float(np.sum(cardinality_value * cardinality_probability))
    theta = (mean_cardinality - eta) / mean_cardinality
    birth_num = max(0, node_num - initial_node_num)
    if birth_num > post_seed_edge_num:
        raise ValueError('PA metadata implies more births than post-seed growth steps.')
    allow_duplicate_edges = bool(parameters.get('allow_duplicate_edges', False))
    rng = np.random.default_rng(seed)
    replicate_num = min(20, max(4, int(math.ceil(theory_sample_num / 10000))))
    r3_sample_per_replicate = max(1000, int(math.ceil(
        theory_sample_num / replicate_num
    )))
    replicate_result = []
    progress_bar = tqdm(
        total = replicate_num,
        desc = f'{metadata["dataset_name"]} BA mean-field integration',
        leave = False, dynamic_ncols = True, unit = 'replicate',
    )
    birth_schedule_attempt_list = []
    for replicate_index in range(replicate_num):
        generated_edge = None
        for birth_schedule_attempt in range(1, 101):
            fixed_cardinality_schedule = (
                rng.permutation(post_seed_cardinality)
                if condition_on_inventory else None
            )
            if birth_num:
                birth_times = np.sort(rng.choice(
                    post_seed_edge_num, size = birth_num, replace = False
                ))
            else:
                birth_times = np.asarray([], dtype = int)
            birth_node = {
                int(birth_time): initial_node_num + index
                for index, birth_time in enumerate(birth_times)
            }

            node_weight = np.zeros(node_num, dtype = float)
            fenwick_tree = np.zeros(node_num + 1, dtype = float)
            seed_denominator = max(total_seed_degree, 1e-12)
            node_weight[:initial_node_num] = (
                seed_degree / np.power(seed_denominator, theta)
            )
            for node_index in range(initial_node_num):
                _func_hm_fenwick_add(
                    fenwick_tree, node_index, node_weight[node_index]
                )

            active_node_num = initial_node_num
            current_edge = []
            occupied = set(seed_edges) if not allow_duplicate_edges else set()
            occupied_cardinality = Counter(len(edge) for edge in occupied)
            schedule_feasible = True

            for creation_time in range(post_seed_edge_num):
                new_node = birth_node.get(creation_time)
                has_birth = new_node is not None
                if condition_on_inventory:
                    cardinality = int(fixed_cardinality_schedule[creation_time])
                    existing_num = cardinality - int(has_birth)
                    if (
                        existing_num > active_node_num
                        or (
                            not allow_duplicate_edges
                            and not has_birth
                            and occupied_cardinality[cardinality]
                            >= math.comb(active_node_num, cardinality)
                        )
                    ):
                        schedule_feasible = False
                        break
                    feasible_index = None
                    feasible_probability = None
                else:
                    feasible_index = []
                    for cardinality_index, cardinality in enumerate(cardinality_value):
                        cardinality = int(cardinality)
                        existing_num = cardinality - int(has_birth)
                        if existing_num > active_node_num:
                            continue
                        if (
                            not allow_duplicate_edges
                            and not has_birth
                            and occupied_cardinality[cardinality]
                            >= math.comb(active_node_num, cardinality)
                        ):
                            continue
                        feasible_index.append(cardinality_index)

                    if not feasible_index:
                        schedule_feasible = False
                        break
                    feasible_probability = cardinality_probability[feasible_index]
                    feasible_probability = (
                        feasible_probability / feasible_probability.sum()
                    )

                selected_edge = None
                while selected_edge is None:
                    if not condition_on_inventory:
                        cardinality_index = int(rng.choice(
                            feasible_index, p = feasible_probability
                        ))
                        cardinality = int(cardinality_value[cardinality_index])
                    existing_num = cardinality - int(has_birth)
                    selected = _func_hm_weighted_without_replacement(
                        fenwick_tree, node_weight, active_node_num,
                        existing_num, rng,
                    )
                    if selected is None:
                        raise ArithmeticError(
                            'PA PPSWOR sampler could not draw from a feasible '
                            'positive-weight node set.'
                        )
                    if has_birth:
                        selected.append(new_node)
                    candidate = tuple(sorted(selected))
                    if allow_duplicate_edges or candidate not in occupied:
                        selected_edge = candidate
                current_edge.append(selected_edge)
                if not allow_duplicate_edges:
                    occupied.add(selected_edge)
                    occupied_cardinality[len(selected_edge)] += 1

                if has_birth:
                    denominator = max(
                        total_seed_degree
                        + mean_cardinality * creation_time,
                        1e-12,
                    )
                    node_weight[new_node] = 1.0 / np.power(
                        denominator, theta
                    )
                    _func_hm_fenwick_add(
                        fenwick_tree, new_node, node_weight[new_node]
                    )
                    active_node_num += 1

            if (
                schedule_feasible
                and len(current_edge) == post_seed_edge_num
                and active_node_num == node_num
            ):
                generated_edge = current_edge
                birth_schedule_attempt_list.append(birth_schedule_attempt)
                break

        if generated_edge is None:
            # 原来在此直接抛出 RuntimeError，中断整个数据集批次：
            # raise RuntimeError(
            #     'Unable to sample a feasible PA mean-field birth schedule after '
            #     '100 attempts.'
            # )
            progress_bar.close()
            reason = (
                'Unable to sample a feasible PA mean-field birth schedule after '
                '100 attempts.'
            )
            tqdm.write(
                f'{metadata["dataset_name"]}: BA theory unavailable; '
                f'replicate {replicate_index + 1}/{replicate_num}. {reason} '
                'Saving NaN probabilities and continuing.'
            )
            # Discard partial integration: it is not a completed theory estimate.
            return {
                r: {
                    'method': 'mean_field_dynamic_support_ppswor_replicate_integration',
                    'exact': False,
                    'available': False,
                    'status': 'skipped_sampling_failure',
                    'reason': reason,
                    'error_type': 'PA_birth_schedule_exhausted',
                    'seed': int(seed),
                    'failed_replicate': int(replicate_index + 1),
                    'completed_replicate_num': int(len(replicate_result)),
                    'replicate_num': int(replicate_num),
                    'attempt_num': 100,
                    'partial_results_discarded': True,
                    'probability': {key: np.nan for key in catalog[r]},
                    'probability_se': {key: np.nan for key in catalog[r]},
                    'probability_ci95': {
                        key: (np.nan, np.nan) for key in catalog[r]
                    },
                    'connected_normalized': False,
                    'assumption': 'Missing theory; exclude from plotting and fit tests.',
                }
                for r in r_values
            }
        if any(len(edge) != len(set(edge)) for edge in generated_edge):
            raise ArithmeticError('PA mean-field sampling produced a degenerate edge.')
        if (
            not allow_duplicate_edges
            and len(set(seed_edges).union(generated_edge))
            != len(seed_edges) + len(generated_edge)
        ):
            raise ArithmeticError('PA mean-field sampling produced duplicate edges.')

        replicate_result.append(alg_hypermotifs_mining(
            generated_edge, r_values = r_values, t = t,
            r3_sample_num = r3_sample_per_replicate,
            seed = _func_hm_seed(
                metadata['dataset_name'], f'ba-theory-{replicate_index}', seed
            ),
            progress_desc = False,
        ))
        progress_bar.update(1)
    progress_bar.close()

    result = {}
    for r in r_values:
        key_list = catalog[r]
        count_matrix = np.asarray([
            [item[r]['counts'].get(key, 0.0) for key in key_list]
            for item in replicate_result
        ], dtype = float)
        total_by_replicate = count_matrix.sum(axis = 1)
        aggregate_count = count_matrix.sum(axis = 0)
        aggregate_total = aggregate_count.sum()
        probability = np.divide(
            aggregate_count, aggregate_total,
            out = np.zeros_like(aggregate_count), where = aggregate_total > 0,
        )
        probability_matrix = np.divide(
            count_matrix,
            total_by_replicate[:, None],
            out = np.zeros_like(count_matrix),
            where = total_by_replicate[:, None] > 0,
        )
        probability_se = (
            np.std(probability_matrix, axis = 0, ddof = 1) / math.sqrt(replicate_num)
            if replicate_num > 1 else np.zeros_like(probability)
        )
        if r == 3:
            sampling_variance = np.asarray([
                [
                    item[r].get('count_se', {}).get(key, 0.0) ** 2
                    for key in key_list
                ]
                for item in replicate_result
            ], dtype = float)
            probability_se = np.sqrt(
                np.square(probability_se)
                + np.sum(sampling_variance, axis = 0)
                / max(aggregate_total, 1.0) ** 2
            )
        result[r] = {
            'method': 'mean_field_dynamic_support_ppswor_replicate_integration',
            'exact': False,
            'probability': dict(zip(key_list, probability.tolist())),
            'probability_se': dict(zip(key_list, probability_se.tolist())),
            'probability_ci95': {
                key: (
                    max(0.0, float(probability[index] - 1.96 * probability_se[index])),
                    min(1.0, float(probability[index] + 1.96 * probability_se[index])),
                )
                for index, key in enumerate(key_list)
            },
            'connected_normalized': True,
            'replicate_num': replicate_num,
            'r3_sample_per_replicate': (
                r3_sample_per_replicate if r == 3 else None
            ),
            'vertex_arrival_probability': eta,
            'condition_on_cardinality_inventory': condition_on_inventory,
            'excluded_seed_edge_num': seed_edge_num,
            'mean_cardinality': mean_cardinality,
            'theta': theta,
            'birth_schedule_attempts': tuple(birth_schedule_attempt_list),
            'assumption': (
                'Mean-field degree trajectories, fixed realized birth inventory, exact '
                'time-varying node support and successive PPSWOR endpoint sampling. '
                'Duplicate suppression matches the dataset generation setting. This is '
                'not an exact finite-time law.'
            ),
        }
    return result


def hypergraph_ci_theory_cal(data_path: str, result_path: str,
                             r_values = (2, 3), t = 6,
                             theory_sample_num = 200000,
                             seed = _HM_DEFAULT_SEED):
    '''计算人工网络对应模型及现实网络三种模型的 connected、unlabeled 理论 C_i。

    BA 出生时间安排耗尽 100 次尝试时保存 NaN 和失败原因，其余模型正常保存。
    此类文件 complete=True 表示流程已处理，status=complete_with_missing 表示
    存在缺失理论；相同输入和参数下再次运行会跳过该文件。
    '''
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['ci'])
    os.makedirs(output_folder, exist_ok = True)
    catalog = _func_hm_load_catalog(result_path, r_values, t)
    settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t),
        'theory_sample_num': int(theory_sample_num),
        'random': 'exact_fixed_inventory',
        'smallworld': 'mixed_cardinality_kernel_monte_carlo',
        'ba': 'mean_field_ppswor_history_monte_carlo',
        'real': 'exact_fixed_inventory_random_baseline',
        'seed': int(seed),
    }
    dataset_list = _func_hm_dataset_list(data_path)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Theory C_i total', dynamic_ncols = True,
        unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        metadata_preview = _func_hm_read_description(dataset_path, dataset_name)
        dataset_settings = dict(settings)
        if metadata_preview['dataset_class'] != 'artificial':
            dataset_settings['real'] = 'observed_parameter_calibrated_r_sw_ba_v1'
        if (
            metadata_preview['dataset_class'] == 'artificial'
            and metadata_preview['network_type'] == 'ba'
        ):
            dataset_settings['ba'] = 'mean_field_dynamic_support_ppswor_v2'
        settings_signature = _func_hm_signature(dataset_settings)
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(output_folder, f'{dataset_name}_ci_theory.pkl')
        manifest_key = f'{dataset_name}:ci-theory'
        if _func_hm_is_complete(
            output_folder, manifest_key, output_path, fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        hyperedge_list, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
        dataset_seed = _func_hm_seed(dataset_name, 'ci-theory', seed)
        network_type = metadata['network_type']
        dataset_class = metadata['dataset_class']
        model_parameter = None
        if dataset_class != 'artificial':
            ci_path = os.path.join(output_folder, f'{dataset_name}_ci.pkl')
            observed_connected_pair_num = None
            if os.path.isfile(ci_path):
                with open(ci_path, 'rb') as file:
                    ci_data = pkl.load(file)
                if 2 in ci_data.get('by_r', {}):
                    observed_connected_pair_num = ci_data['by_r'][2].get(
                        'total_count'
                    )
            real_input = _func_hm_real_model_inputs(
                metadata['node_num'], hyperedge_list, metadata, t,
                observed_connected_pair_num = observed_connected_pair_num,
            )
            eligible_edges = real_input['eligible_edges']
            random_by_r = _func_hm_random_theory(
                metadata['node_num'], eligible_edges, catalog, r_values,
                t, real_input['random_allow_duplicate_edges'],
            )
            if eligible_edges:
                smallworld_by_r = _func_hm_smallworld_theory(
                    metadata['node_num'], eligible_edges,
                    real_input['smallworld_metadata'], catalog,
                    r_values, theory_sample_num,
                    _func_hm_seed(dataset_name, 'ci-theory-real-sw', seed),
                )
            else:
                smallworld_by_r = _func_hm_empty_theory(
                    catalog, r_values, 'empty_smallworld_model',
                    'The real network has no hyperedges in the selected cardinality range.',
                )
            if len(real_input['pa_reference_edges']) >= 2:
                pa_by_r = _func_hm_pa_meanfield_theory(
                    metadata['node_num'], real_input['pa_reference_edges'],
                    real_input['pa_metadata'], catalog, r_values,
                    theory_sample_num, t,
                    _func_hm_seed(dataset_name, 'ci-theory-real-ba', seed),
                )
            else:
                pa_by_r = _func_hm_empty_theory(
                    catalog, r_values, 'empty_preferential_attachment_model',
                    'At least two eligible hyperedges are required for motif comparison.',
                )
            by_model = {
                'r': random_by_r,
                'sw': smallworld_by_r,
                'ba': pa_by_r,
            }
            # Keep the former field as the random baseline for old readers.
            by_r = random_by_r
            baseline = 'real_network_observed_parameter_r_sw_ba_comparison'
            model_parameter = real_input['model_input']
        elif network_type == 'r':
            allow_duplicate = bool(
                metadata.get('parameters', {}).get('allow_duplicate_edges', False)
            )
            by_r = _func_hm_random_theory(
                metadata['node_num'], hyperedge_list, catalog, r_values,
                t, allow_duplicate,
            )
            baseline = 'random_exact'
            by_model = {'r': by_r}
        elif network_type == 'sw':
            by_r = _func_hm_smallworld_theory(
                metadata['node_num'], hyperedge_list, metadata, catalog,
                r_values, theory_sample_num, dataset_seed,
            )
            baseline = 'smallworld_mixed_cardinality_kernel'
            by_model = {'sw': by_r}
        elif network_type == 'ba':
            by_r = _func_hm_pa_meanfield_theory(
                metadata['node_num'], hyperedge_list, metadata, catalog,
                r_values, theory_sample_num, t, dataset_seed,
            )
            baseline = 'preferential_attachment_mean_field'
            by_model = {'ba': by_r}
        else:
            by_r = _func_hm_random_theory(
                metadata['node_num'], hyperedge_list, catalog, r_values,
                t, False,
            )
            baseline = 'random_exact_fallback'
            by_model = {'r': by_r}
        result = _func_hm_result_header(
            dataset_name, fingerprint, dataset_settings, metadata, dataset_seed
        )
        unavailable_models = {
            model_type: {
                r: item.get('reason', 'Theory unavailable')
                for r, item in model_result.items()
                if item.get('available') is False
            }
            for model_type, model_result in by_model.items()
            if any(item.get('available') is False for item in model_result.values())
        }
        result.update({
            'by_r': by_r,
            'by_model': by_model,
            # complete means processed, including explicitly recorded missing models.
            'status': 'complete_with_missing' if unavailable_models else 'complete',
            'all_models_available': not bool(unavailable_models),
            'unavailable_models': unavailable_models,
            'missing_theory_policy': 'NaN; skip fit tests and plotting; cache outcome',
            'baseline': baseline,
            'model_parameter': model_parameter,
            'unlabeled': True,
            'connected_normalized': True,
            'validation': None,
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, manifest_key, fingerprint, settings_signature
        )
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def hypergraph_deltai_cal(data_path: str, result_path: str,
                          r_values = (2, 3), t = 6, epsilon = 4.0,
                          seed = _HM_DEFAULT_SEED):
    '''使用原网络与 100 个 null models 的 motif 数量计算 Delta_i。'''
    motif_folder = os.path.join(result_path, _HM_RESULT_FOLDER['motif'])
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['deltai'])
    os.makedirs(output_folder, exist_ok = True)
    catalog = _func_hm_load_catalog(result_path, r_values, t)
    settings = {
        'r_values': tuple(r_values), 'maximum_cardinality': int(t),
        'epsilon': float(epsilon), 'seed': int(seed),
    }
    settings_signature = _func_hm_signature(settings)
    dataset_list = _func_hm_dataset_list(data_path)
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Delta_i total', dynamic_ncols = True, unit = 'network'
    )
    for dataset_index, (dataset_name, dataset_path) in enumerate(outer_bar, start = 1):
        fingerprint = _func_hm_source_fingerprint(dataset_path)
        output_path = os.path.join(output_folder, f'{dataset_name}_deltai.pkl')
        manifest_key = f'{dataset_name}:deltai'
        if _func_hm_is_complete(
            output_folder, manifest_key, output_path, fingerprint, settings_signature
        ):
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        original_path = os.path.join(
            motif_folder, f'{dataset_name}_hypermotifs.pkl'
        )
        null_path = os.path.join(
            motif_folder, f'{dataset_name}_null_model_hypermotifs.pkl'
        )
        if not os.path.isfile(original_path) or not os.path.isfile(null_path):
            raise FileNotFoundError(f'Missing motif inputs for Delta_i: {dataset_name}')
        with open(original_path, 'rb') as file:
            original_data = pkl.load(file)
        with open(null_path, 'rb') as file:
            null_data = pkl.load(file)

        by_r = {}
        for r in r_values:
            motif_key_list = catalog[r]
            real_counts = np.asarray([
                original_data['by_r'][r]['counts'].get(key, 0.0)
                for key in motif_key_list
            ], dtype = float)
            real_se = np.asarray([
                original_data['by_r'][r].get('count_se', {}).get(key, 0.0)
                for key in motif_key_list
            ], dtype = float)
            null_count_matrix = np.asarray([
                [
                    item['by_r'][r]['counts'].get(key, 0.0)
                    for key in motif_key_list
                ]
                for item in null_data['null_results']
            ], dtype = float)
            null_sampling_se_matrix = np.asarray([
                [
                    item['by_r'][r].get('count_se', {}).get(key, 0.0)
                    for key in motif_key_list
                ]
                for item in null_data['null_results']
            ], dtype = float)
            null_mean = np.mean(null_count_matrix, axis = 0)
            null_std = (
                np.std(null_count_matrix, axis = 0, ddof = 1)
                if null_count_matrix.shape[0] > 1
                else np.zeros_like(null_mean)
            )
            null_mean_se = null_std / math.sqrt(max(1, null_count_matrix.shape[0]))
            if null_sampling_se_matrix.size:
                null_mean_se = np.sqrt(
                    np.square(null_mean_se)
                    + np.sum(np.square(null_sampling_se_matrix), axis = 0)
                    / max(1, null_count_matrix.shape[0]) ** 2
                )
            denominator = real_counts + null_mean + float(epsilon)
            delta = np.divide(
                real_counts - null_mean,
                denominator,
                out = np.zeros_like(denominator),
                where = denominator > 0,
            )
            derivative_real = (2.0 * null_mean + float(epsilon)) / np.square(denominator)
            derivative_null = -(2.0 * real_counts + float(epsilon)) / np.square(denominator)
            delta_se = np.sqrt(
                np.square(derivative_real * real_se)
                + np.square(derivative_null * null_mean_se)
            )
            delta = np.clip(delta, -1.0, 1.0)
            by_r[r] = {
                'motif_keys': motif_key_list,
                'real_counts': dict(zip(motif_key_list, real_counts.tolist())),
                'real_count_se': dict(zip(motif_key_list, real_se.tolist())),
                'null_counts': {
                    key: null_count_matrix[:, key_index].tolist()
                    for key_index, key in enumerate(motif_key_list)
                },
                'null_mean': dict(zip(motif_key_list, null_mean.tolist())),
                'null_std': dict(zip(motif_key_list, null_std.tolist())),
                'null_mean_se': dict(zip(motif_key_list, null_mean_se.tolist())),
                'deltai': dict(zip(motif_key_list, delta.tolist())),
                'deltai_se': dict(zip(motif_key_list, delta_se.tolist())),
                'deltai_ci95': {
                    key: (
                        max(-1.0, float(delta[key_index] - 1.96 * delta_se[key_index])),
                        min(1.0, float(delta[key_index] + 1.96 * delta_se[key_index])),
                    )
                    for key_index, key in enumerate(motif_key_list)
                },
            }
        result = _func_hm_result_header(
            dataset_name, fingerprint, settings, original_data['metadata'],
            _func_hm_seed(dataset_name, 'deltai', seed),
        )
        result.update({
            'by_r': by_r,
            'null_num': len(null_data['null_results']),
            'epsilon': float(epsilon),
        })
        _func_hm_atomic_pickle(result, output_path)
        _func_hm_mark_complete(
            output_folder, manifest_key, fingerprint, settings_signature
        )
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()


def _func_hm_safe_correlation(left, right, method = 'pearson'):
    left, right = np.asarray(left, dtype = float), np.asarray(right, dtype = float)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return {'statistic': np.nan, 'pvalue': np.nan}
    if method == 'pearson':
        statistic, p_value = stats.pearsonr(left, right)
    else:
        statistic, p_value = stats.spearmanr(left, right)
    return {'statistic': float(statistic), 'pvalue': float(p_value)}


def _func_hm_g_test(observed, theoretical_probability):
    observed = np.asarray(observed, dtype = float)
    theoretical_probability = np.asarray(theoretical_probability, dtype = float)
    total = observed.sum()
    expected = theoretical_probability * total
    regular = expected >= 5.0
    pooled_observed = observed[regular].tolist()
    pooled_expected = expected[regular].tolist()
    if np.any(~regular):
        pooled_observed.append(float(observed[~regular].sum()))
        pooled_expected.append(float(expected[~regular].sum()))
    pooled_observed = np.asarray(pooled_observed, dtype = float)
    pooled_expected = np.asarray(pooled_expected, dtype = float)
    positive = pooled_expected > 0
    pooled_observed, pooled_expected = pooled_observed[positive], pooled_expected[positive]
    if pooled_observed.size < 2 or pooled_observed.sum() <= 0:
        return {'statistic': np.nan, 'pvalue': np.nan, 'bin_num': int(pooled_observed.size)}
    pooled_expected *= pooled_observed.sum() / pooled_expected.sum()
    statistic, p_value = stats.power_divergence(
        pooled_observed, pooled_expected, lambda_ = 'log-likelihood'
    )
    return {
        'statistic': float(statistic),
        'pvalue': float(p_value),
        'bin_num': int(pooled_observed.size),
    }


def _func_hm_bootstrap_fit(empirical, theoretical, effective_sample_size,
                           bootstrap_num, rng):
    empirical = np.asarray(empirical, dtype = float)
    theoretical = np.asarray(theoretical, dtype = float)
    if empirical.sum() <= 0 or theoretical.sum() <= 0:
        return {'statistic': np.nan, 'pvalue': np.nan, 'bootstrap_num': 0}
    empirical /= empirical.sum()
    theoretical /= theoretical.sum()
    observed_statistic = float(sp.distance.jensenshannon(
        empirical, theoretical, base = 2
    ))
    sample_num = int(max(10, min(1000000, round(effective_sample_size or 10))))
    simulated_statistic = np.empty(int(bootstrap_num), dtype = float)
    for bootstrap_index in range(int(bootstrap_num)):
        simulated = rng.multinomial(sample_num, theoretical) / sample_num
        simulated_statistic[bootstrap_index] = sp.distance.jensenshannon(
            simulated, theoretical, base = 2
        )
    p_value = (
        1 + np.count_nonzero(simulated_statistic >= observed_statistic)
    ) / (int(bootstrap_num) + 1)
    return {
        'statistic': observed_statistic,
        'pvalue': float(p_value),
        'bootstrap_num': int(bootstrap_num),
        'effective_sample_size': sample_num,
    }


def hypergraph_ci_theory_validate(data_path: str, result_path: str,
                                  r_values = (2, 3), t = 6,
                                  bootstrap_num = 1000,
                                  seed = _HM_DEFAULT_SEED):
    '''验证实际 C_i 与对应理论 connected distribution 的符合程度。'''
    ci_folder = os.path.join(result_path, _HM_RESULT_FOLDER['ci'])
    catalog = _func_hm_load_catalog(result_path, r_values, t)
    dataset_list = _func_hm_dataset_list(data_path)
    summary = {}
    start_time = time.time()
    outer_bar = tqdm(
        dataset_list, desc = 'Theory validation total', dynamic_ncols = True,
        unit = 'network'
    )
    for dataset_index, (dataset_name, _dataset_path) in enumerate(outer_bar, start = 1):
        ci_path = os.path.join(ci_folder, f'{dataset_name}_ci.pkl')
        theory_path = os.path.join(ci_folder, f'{dataset_name}_ci_theory.pkl')
        if not os.path.isfile(ci_path) or not os.path.isfile(theory_path):
            raise FileNotFoundError(f'Missing C_i/theory result for {dataset_name}.')
        with open(ci_path, 'rb') as file:
            ci_data = pkl.load(file)
        with open(theory_path, 'rb') as file:
            theory_data = pkl.load(file)
        validation_settings = {
            'r_values': tuple(r_values),
            'maximum_cardinality': int(t),
            'bootstrap_num': int(bootstrap_num),
            'seed': int(seed),
            'ci_settings_signature': ci_data.get('settings_signature'),
            'theory_settings_signature': theory_data.get('settings_signature'),
        }
        validation_signature = _func_hm_signature(validation_settings)
        if (
            theory_data.get('validation_by_model') is not None
            and theory_data.get('validation_settings_signature') == validation_signature
        ):
            summary[dataset_name] = {
                'metadata': theory_data['metadata'],
                'baseline': theory_data['baseline'],
                'by_r': theory_data['validation'],
                'by_model': theory_data['validation_by_model'],
            }
            outer_bar.set_postfix_str(f'{dataset_name}: skipped')
            continue
        comparison_by_r = ci_data.get('theory_comparison_by_r') or ci_data['by_r']
        default_model = (
            theory_data['metadata'].get('network_type', 'r')
            if theory_data['metadata'].get('dataset_class') == 'artificial'
            else 'r'
        )
        theory_by_model = theory_data.get('by_model') or {
            default_model: theory_data['by_r']
        }
        validation_by_model = {}
        for model_type, theory_by_r in theory_by_model.items():
            validation = {}
            rng = np.random.default_rng(_func_hm_seed(
                dataset_name, f'fit-bootstrap-{model_type}', seed
            ))
            for r in r_values:
                key_list = catalog[r]
                theory_item = theory_by_r[r]
                if theory_item.get('available') is False:
                    validation[r] = {
                        'status': 'skipped_missing_theory',
                        'reason': theory_item.get('reason', 'Theory unavailable'),
                        'jensen_shannon_distance': np.nan,
                        'total_variation_distance': np.nan,
                        'pearson': {'statistic': np.nan, 'pvalue': np.nan},
                        'spearman': {'statistic': np.nan, 'pvalue': np.nan},
                        'goodness_of_fit': {
                            'method': 'not_performed_missing_theory',
                            'statistic': np.nan,
                            'pvalue': np.nan,
                        },
                    }
                    continue
                empirical = np.asarray([
                    comparison_by_r[r]['ci'].get(key, 0.0) for key in key_list
                ], dtype = float)
                theoretical = np.asarray([
                    theory_by_r[r]['probability'].get(key, 0.0)
                    for key in key_list
                ], dtype = float)
                if empirical.sum() > 0:
                    empirical /= empirical.sum()
                if theoretical.sum() > 0:
                    theoretical /= theoretical.sum()
                js_distance = (
                    float(sp.distance.jensenshannon(
                        empirical, theoretical, base = 2
                    ))
                    if empirical.sum() > 0 and theoretical.sum() > 0
                    else np.nan
                )
                item = {
                    'jensen_shannon_distance': js_distance,
                    'total_variation_distance': float(
                        0.5 * np.abs(empirical - theoretical).sum()
                    ),
                    'pearson': _func_hm_safe_correlation(
                        empirical, theoretical, 'pearson'
                    ),
                    'spearman': _func_hm_safe_correlation(
                        empirical, theoretical, 'spearman'
                    ),
                }
                if r == 2:
                    observed_count = [
                        comparison_by_r[r]['counts'].get(key, 0.0)
                        for key in key_list
                    ]
                    item['goodness_of_fit'] = {
                        'method': 'pooled_likelihood_ratio_g_test',
                        **_func_hm_g_test(observed_count, theoretical),
                    }
                else:
                    item['goodness_of_fit'] = {
                        'method': 'effective_sample_size_parametric_bootstrap',
                        **_func_hm_bootstrap_fit(
                            empirical, theoretical,
                            comparison_by_r[r].get('effective_sample_size'),
                            bootstrap_num, rng,
                        ),
                    }
                validation[r] = item
            validation_by_model[model_type] = validation
        primary_model = (
            default_model if default_model in validation_by_model
            else next(iter(validation_by_model))
        )
        validation = validation_by_model[primary_model]
        theory_data['validation'] = validation
        theory_data['validation_by_model'] = validation_by_model
        theory_data['validation_settings'] = validation_settings
        theory_data['validation_settings_signature'] = validation_signature
        theory_data['validation_updated_at'] = datetime.now().isoformat(timespec = 'seconds')
        _func_hm_atomic_pickle(theory_data, theory_path)
        summary[dataset_name] = {
            'metadata': theory_data['metadata'],
            'baseline': theory_data['baseline'],
            'by_r': validation,
            'by_model': validation_by_model,
        }
        outer_bar.set_postfix_str(_func_hm_progress_text(
            start_time, dataset_index, len(dataset_list), dataset_name
        ))
    outer_bar.close()
    _func_hm_atomic_pickle({
        'schema_version': _HM_SCHEMA_VERSION,
        'complete': True,
        'settings': {
            'r_values': tuple(r_values),
            'maximum_cardinality': int(t),
            'bootstrap_num': int(bootstrap_num),
            'seed': int(seed),
        },
        'settings_signature': _func_hm_signature({
            'r_values': tuple(r_values),
            'maximum_cardinality': int(t),
            'bootstrap_num': int(bootstrap_num),
            'seed': int(seed),
        }),
        'source_fingerprint': 'per-dataset fingerprints stored in by_dataset',
        'seed': int(seed),
        'bootstrap_num': int(bootstrap_num),
        'by_dataset': summary,
        'created_at': datetime.now().isoformat(timespec = 'seconds'),
    }, os.path.join(ci_folder, 'ci_theory_validation_summary.pkl'))


def _func_hm_holm_adjust(p_value_list):
    result = [np.nan] * len(p_value_list)
    finite_index = [index for index, value in enumerate(p_value_list) if np.isfinite(value)]
    ordered = sorted(finite_index, key = lambda index: p_value_list[index])
    previous = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * p_value_list[index])
        adjusted = max(previous, adjusted)
        result[index] = adjusted
        previous = adjusted
    return result


def hypergraph_ar_significance_cal(data_path: str, result_path: str,
                                   r_values = (2, 3), t = 6):
    '''计算人工与现实网络 C_i、Delta_i 分布的双侧 Mann-Whitney U 检验。'''
    ci_folder = os.path.join(result_path, _HM_RESULT_FOLDER['ci'])
    delta_folder = os.path.join(result_path, _HM_RESULT_FOLDER['deltai'])
    output_folder = os.path.join(result_path, _HM_RESULT_FOLDER['compare_fig'])
    os.makedirs(output_folder, exist_ok = True)
    output_path = os.path.join(output_folder, '0.significance_test.pkl')
    dataset_list = _func_hm_dataset_list(data_path)
    input_fingerprint = {
        dataset_name: _func_hm_source_fingerprint(dataset_path)
        for dataset_name, dataset_path in dataset_list
    }
    settings = {
        'r_values': tuple(r_values),
        'maximum_cardinality': int(t),
        'method': 'two-sided Mann-Whitney U with Holm correction',
        'input_fingerprint': input_fingerprint,
    }
    settings_signature = _func_hm_signature(settings)
    if os.path.isfile(output_path):
        try:
            with open(output_path, 'rb') as file:
                old_result = pkl.load(file)
            if (
                old_result.get('complete')
                and old_result.get('settings_signature') == settings_signature
            ):
                return
        except (OSError, EOFError, pkl.UnpicklingError):
            pass
    catalog = _func_hm_load_catalog(result_path, r_values, t)
    value = {
        metric_name: {
            r: {'artificial': [], 'real': []} for r in r_values
        }
        for metric_name in ['ci', 'deltai']
    }
    progress_bar = tqdm(
        dataset_list, desc = 'Significance data total', dynamic_ncols = True,
        unit = 'network'
    )
    for dataset_name, dataset_path in progress_bar:
        metadata = _func_hm_read_description(dataset_path, dataset_name)
        group = 'artificial' if metadata['dataset_class'] == 'artificial' else 'real'
        ci_path = os.path.join(ci_folder, f'{dataset_name}_ci.pkl')
        delta_path = os.path.join(delta_folder, f'{dataset_name}_deltai.pkl')
        if not os.path.isfile(ci_path) or not os.path.isfile(delta_path):
            continue
        with open(ci_path, 'rb') as file:
            ci_data = pkl.load(file)
        with open(delta_path, 'rb') as file:
            delta_data = pkl.load(file)
        for r in r_values:
            value['ci'][r][group].extend(
                ci_data['by_r'][r]['ci'].get(key, 0.0) for key in catalog[r]
            )
            value['deltai'][r][group].extend(
                delta_data['by_r'][r]['deltai'].get(key, 0.0) for key in catalog[r]
            )
    progress_bar.close()

    tests, raw_p = [], []
    for metric_name in ['ci', 'deltai']:
        for r in r_values:
            artificial = value[metric_name][r]['artificial']
            real = value[metric_name][r]['real']
            if artificial and real:
                statistic, p_value = stats.mannwhitneyu(
                    artificial, real, alternative = 'two-sided'
                )
                status = 'complete'
            else:
                statistic, p_value, status = np.nan, np.nan, 'insufficient_real_data'
            tests.append({
                'metric': metric_name,
                'r': r,
                'method': 'two-sided Mann-Whitney U',
                'statistic': float(statistic),
                'pvalue': float(p_value),
                'artificial_value_num': len(artificial),
                'real_value_num': len(real),
                'status': status,
            })
            raw_p.append(float(p_value))
    adjusted = _func_hm_holm_adjust(raw_p)
    for item, adjusted_p in zip(tests, adjusted):
        item['holm_adjusted_pvalue'] = adjusted_p
    _func_hm_atomic_pickle({
        'schema_version': _HM_SCHEMA_VERSION,
        'complete': True,
        'settings': settings,
        'settings_signature': settings_signature,
        'source_fingerprint': input_fingerprint,
        'seed': None,
        'tests': tests,
        'created_at': datetime.now().isoformat(timespec = 'seconds'),
    }, output_path)


def hypergraph_analysis_self_check():
    '''返回小型确定性结构检查结果；该方法不会由 main.py 自动执行。'''
    hyperedge_list = [(0, 1), (1, 2), (0, 1, 2)]
    r2_result = _func_hm_r2_exact(hyperedge_list, t = 3)
    atom = _func_hm_atom_vector(tuple(hyperedge_list))
    canonical = _func_hm_canonical_atom(atom)
    permutation_invariant = all(
        _func_hm_canonical_atom(_func_hm_permute_atom(atom, permutation)) == canonical
        for permutation in permutations(range(3))
    )
    state = _func_hm_build_curveball_state(hyperedge_list, 3)
    degree = np.asarray([len(item) for item in state['node_edges']], dtype = np.int64)
    cardinality = np.asarray([len(edge) for edge in hyperedge_list], dtype = np.int64)
    validation = _func_hm_validate_curveball(
        state, degree, cardinality, int(cardinality.sum())
    )
    return {
        'r2_total_is_three': r2_result['total'] == 3,
        'r3_label_permutation_invariant': permutation_invariant,
        'curveball_initial_state_valid': all(
            validation[key] for key in [
                'node_degree_exact', 'cardinality_exact', 'incidence_exact',
                'no_degenerate_hyperedge'
            ]
        ),
    }


# ------------------------连通超图 N / M / Delta 生成------------------------

@lru_cache(maxsize = 8)
def _func_hm_generation_keys(max_r, max_v):
    catalog = _func_hm_catalog(tuple(range(2, max_r + 1)), max_v)
    return tuple(key for r in range(2, max_r + 1) for key in catalog[r])


@lru_cache(maxsize = 65536)
def _func_hm_generation_canonical(atom):
    return _func_hm_canonical_atom(atom)


def _func_hm_generation_connected(edges, node_num):
    '''覆盖全部 N 个节点并连通；不能忽略孤立节点。'''
    incident = defaultdict(list)
    for index, edge in enumerate(edges):
        if len(edge) < 2 or len(edge) != len(set(edge)):
            return False
        for node in edge:
            if node < 0 or node >= node_num:
                return False
            incident[node].append(index)
    if len(incident) != node_num:
        return False
    seen, seen_edges, stack = {0}, set(), [0]
    while stack:
        for index in incident[stack.pop()]:
            if index in seen_edges:
                continue
            seen_edges.add(index)
            for node in edges[index]:
                if node not in seen:
                    seen.add(node)
                    stack.append(node)
    return len(seen) == node_num


def _func_hm_generation_initial(node_num, edge_num, max_v, rng):
    '''每条新边连接已访问节点，同时预留覆盖剩余节点的容量。'''
    capacity = min(node_num, max_v)
    if node_num < 2 or edge_num < 1 or node_num > 1 + edge_num * (capacity - 1):
        raise ValueError('N/M/max_v cannot support a connected hypergraph on all N nodes.')
    labels = [int(node) for node in rng.permutation(node_num)]
    visited, position, edges = [labels[0]], 1, []
    for index in range(edge_num):
        remaining = node_num - position
        minimum = max(0, remaining - (edge_num - index - 1) * (capacity - 1))
        if len(visited) == 1:
            minimum = max(1, minimum)
        added = int(rng.integers(minimum, min(capacity - 1, remaining) + 1))
        old_num = int(rng.integers(max(1, 2 - added), min(len(visited), capacity - added) + 1))
        old_nodes = [int(v) for v in rng.choice(visited, old_num, replace = False)]
        new_nodes = labels[position:position + added]
        edges.append(tuple(sorted(old_nodes + new_nodes)))
        visited.extend(new_nodes)
        position += added
    if position != node_num or not _func_hm_generation_connected(edges, node_num):
        raise AssertionError('Connected construction failed its coverage invariant.')
    return edges


def _func_hm_generation_target(Delta_i, keys, max_r):
    if isinstance(Delta_i, dict) and 'by_r' in Delta_i:
        merged = {}
        for r in range(2, max_r + 1):
            if r not in Delta_i['by_r']:
                raise ValueError(f'The input Delta result has no r={r} data.')
            merged.update(Delta_i['by_r'][r]['deltai'])
        Delta_i = merged
    if isinstance(Delta_i, dict):
        if set(Delta_i) - set(keys):
            raise ValueError('Delta_i contains motif keys outside the requested canonical catalogue.')
        values = np.asarray([float(Delta_i.get(key, 0.0)) for key in keys], dtype = float)
    else:
        values = np.asarray(Delta_i, dtype = float)
    if values.shape != (len(keys),):
        raise ValueError(f'Delta_i must be a one-dimensional vector of length {len(keys)}.')
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError('Delta_i entries must be finite and within [-1, 1].')
    return values.copy()


def _func_hm_generation_counts(edges, keys, max_r, max_v, seed, sample_num, exact_limit):
    '''复用统计器；小型 r=3 用已有 Venn 工具枚举，消除不必要的抽样噪声。'''
    # Compiled exact counts remove rare-motif sampling noise in the practical
    # reconstruction range. exact_limit=0 still explicitly requests sampling.
    if exact_limit > 0 and len(edges) <= 1536:
        from delta_reconstruction import exact_counts
        return exact_counts(edges, tuple(keys), max_r)
    if len(edges) > 1536 and (max_r == 2 or math.comb(len(edges), 3) > exact_limit):
        from delta_reconstruction_dense import applicable, counts
        node_num = 1 + max(node for edge in edges for node in edge)
        if applicable(node_num, len(edges)):
            try:
                return counts(edges, node_num, tuple(keys), max_r, sample_num, seed)
            except MemoryError:
                # Retain the existing sparse estimator if projected adjacency
                # would exceed the dense backend's explicit memory bound.
                pass
    result = alg_hypermotifs_mining(edges, r_values = (2,), t = max_v)
    count_map = dict(result[2]['counts'])
    if max_r == 3:
        if math.comb(len(edges), 3) <= exact_limit:
            triples = Counter()
            for selected in combinations(edges, 3):
                atom = _func_hm_atom_vector(selected)
                if _func_hm_atom_connected(atom):
                    triples[(3,) + _func_hm_generation_canonical(atom)] += 1
            count_map.update(triples)
        else:
            sampled = alg_hypermotifs_mining(
                edges, r_values = (3,), t = max_v, r3_sample_num = sample_num,
                seed = seed, progress_desc = False,
            )
            count_map.update(sampled[3]['counts'])
    return np.asarray([count_map.get(key, 0.0) for key in keys], dtype = float)


def _func_hm_generation_null_mean(edges, node_num, keys, max_r, max_v,
                                   null_num, burn_in, seed, sample_num, exact_limit,
                                   description, verbose, deadline = None):
    '''实际生成并验证每个 null，统计后释放；不额外限制 null 连通性。'''
    original = _func_hm_build_curveball_state(edges, node_num)
    degree = np.asarray([len(item) for item in original['node_edges']], dtype = np.int64)
    cardinality = np.asarray([len(edge) for edge in edges], dtype = np.int64)
    total = np.zeros(len(keys), dtype = float)
    rng = np.random.default_rng(seed)
    start = time.time()
    with tqdm(total = null_num, desc = description, unit = 'null', leave = False,
              dynamic_ncols = True, disable = not verbose) as progress:
        for index in range(null_num):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('Candidate scoring reached the search time budget.')
            if node_num * len(edges) <= 20000000:
                from delta_reconstruction import pack, unpack, curveball
                table, sizes = pack(edges)
                randomizer = curveball
                if len(edges) > 1536:
                    from delta_reconstruction_dense import applicable, edge_curveball
                    if applicable(node_num, len(edges)):
                        randomizer = edge_curveball
                randomized = randomizer(table, sizes, node_num, burn_in,
                                        int(rng.integers(2 ** 31)))
                state = _func_hm_build_curveball_state(unpack(randomized, sizes), node_num)
            else:
                state = _func_hm_copy_curveball_state(original)
                for offset in range(0, burn_in, 1000):
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise TimeoutError('Candidate scoring reached the search time budget.')
                    _func_hm_curveball_trades(state, min(1000, burn_in - offset), rng)
            _func_hm_validate_curveball(state, degree, cardinality, int(cardinality.sum()))
            null_edges = [tuple(sorted(edge)) for edge in state['edge_nodes']]
            total += _func_hm_generation_counts(
                null_edges, keys, max_r, max_v, int(rng.integers(2 ** 31)), sample_num, exact_limit,
            )
            progress.update(1)
            if verbose:
                progress.set_postfix_str(_func_hm_progress_text(
                    start, index + 1, null_num, f'N={node_num}, M={len(edges)}'
                ), refresh = False)
    return total / null_num


def _func_hm_generation_delta(counts, null_mean):
    return (counts - null_mean) / (counts + null_mean + 4.0)


def _func_hm_generation_proposal(edges, node_num, max_v, rng):
    proposal = list(edges)
    if rng.random() < 0.65:
        state = _func_hm_build_curveball_state(edges, node_num)
        _func_hm_curveball_trades(state, int(rng.integers(1, 5)), rng)
        proposal = [tuple(sorted(edge)) for edge in state['edge_nodes']]
    else:
        index = int(rng.integers(len(edges)))
        edge = set(edges[index])
        choice = int(rng.integers(3))
        available = [node for node in range(node_num) if node not in edge]
        if choice == 0 and available:
            edge.remove(int(rng.choice(sorted(edge))))
            edge.add(int(rng.choice(available)))
        elif choice == 1 and available and len(edge) < max_v:
            edge.add(int(rng.choice(available)))
        elif choice == 2 and len(edge) > 2:
            edge.remove(int(rng.choice(sorted(edge))))
        proposal[index] = tuple(sorted(edge))
    return proposal if _func_hm_generation_connected(proposal, node_num) else None


def _func_hm_generation_search(initial, node_num, target, keys, max_r, max_v,
                                seed, max_iter, max_seconds, tolerance,
                                search_null_num, sample_num, exact_limit, verbose):
    '''按阶数及非零幅度平衡损失；外层改边际，内层精确更新 motif。'''
    if len(initial) <= 1536 and node_num * len(initial) <= 20000000:
        from delta_reconstruction import search, null_mean
        def estimator(edges, samples, null_seed, deadline):
            return null_mean(edges, node_num, tuple(keys), max_r, samples,
                             max(100, 1000*len(edges)), null_seed, deadline)
        generated, _ = search(initial, node_num, target, tuple(keys), max_r, max_v,
                              seed, max_iter, max_seconds, tolerance,
                              search_null_num, estimator, verbose)
        return generated
    from delta_reconstruction_dense import applicable, search as dense_search
    if applicable(node_num, len(initial)):
        generated, _ = dense_search(initial, node_num, target, tuple(keys), max_r, max_v,
                                    seed, max_iter, max_seconds, tolerance,
                                    search_null_num, sample_num, verbose)
        return generated
    return _func_hm_generation_search_sampled(
        initial, node_num, target, keys, max_r, max_v, seed, max_iter,
        max_seconds, tolerance, search_null_num, sample_num, exact_limit, verbose)


def _func_hm_generation_search_sampled(initial, node_num, target, keys, max_r, max_v,
                                seed, max_iter, max_seconds, tolerance,
                                search_null_num, sample_num, exact_limit, verbose):
    '''有限预算搜索；Delta 未达标仍返回已验证连通的最佳候选。'''
    rng = np.random.default_rng(seed)
    started, deadline = time.time(), time.perf_counter() + max_seconds
    current, best = list(initial), list(initial)
    from delta_reconstruction import loss_weights
    weights = loss_weights(target, keys)
    current_loss, best_loss, cache = float('inf'), float('inf'), {}
    with tqdm(total = max_iter + 1, desc = 'Connected Delta search', unit = 'candidate',
              dynamic_ncols = True, disable = not verbose) as progress:
        for iteration in range(max_iter + 1):
            if time.perf_counter() >= deadline:
                break
            if iteration == 0:
                candidate = current
            elif iteration % 20 == 0:
                candidate = _func_hm_generation_initial(node_num, len(initial), max_v, rng)
            else:
                candidate = _func_hm_generation_proposal(current, node_num, max_v, rng)
            progress.update(1)
            if candidate is None:
                continue
            degree = Counter(node for edge in candidate for node in edge)
            # 此近似缓存仅用于搜索；最终均值总是基于最终图重新独立计算。
            margin = (tuple(sorted(degree.values())), tuple(sorted(map(len, candidate))))
            try:
                if margin not in cache:
                    mean = _func_hm_generation_null_mean(
                        candidate, node_num, keys, max_r, max_v, search_null_num,
                        max(100, 1000 * len(candidate)), int(rng.integers(2 ** 31)),
                        sample_num, exact_limit, 'Search nulls', False, deadline,
                    )
                    if len(cache) >= 32:
                        cache.pop(next(iter(cache)))
                    cache[margin] = mean
                counts = _func_hm_generation_counts(
                    candidate, keys, max_r, max_v, int(rng.integers(2 ** 31)),
                    sample_num, exact_limit,
                )
            except TimeoutError:
                break
            difference = np.abs(_func_hm_generation_delta(counts, cache[margin]) - target)
            maximum, rmse = float(difference.max()), float(np.sqrt(np.mean(difference ** 2)))
            loss = float(np.dot(weights, difference ** 2))
            if loss < best_loss:
                best, best_loss = list(candidate), loss
            temperature = max(0.002, 0.05 * (1.0 - iteration / max(1, max_iter)))
            if loss <= current_loss or rng.random() < math.exp(min(0.0, (current_loss - loss) / temperature)):
                current, current_loss = list(candidate), loss
            if verbose:
                progress.set_postfix_str(_func_hm_progress_text(
                    started, iteration + 1, max_iter + 1, f'N={node_num}, M={len(initial)}'
                ), refresh = False)
            if maximum <= tolerance:
                break
    return best


_HM_GENERATION_SAVE_LOCK = threading.Lock()


def hypergraph_generation_model_N_M_delta(save_path, mode, N = None, M = None,
                                           Delta_i = None, max_r = 3, max_v = 6,
                                           null_num = 100, *, seed = None,
                                           tolerance = 0.05, max_iter = 5000000,
                                           max_seconds = 3600.0, search_null_num = 10,
                                           burn_in = None, r3_sample_num = 200000,
                                           exact_limit = 20000, verbose = True,
                                           output_folder = None):
    '''生成覆盖全部 N 个节点的连通超图，并保存恰好五个字段。

    mode='random'：缺失 N 时抽取小规模值；缺失 M 时从约0.4N..1.2N的
    可行范围抽取。缺失 Delta_i 时，用聚集强度随机变化的独立连通参考图
    及其 null 均值产生目标，再重新构造候选进行搜索。
    参考图不直接作为最终候选；最终 null 随机流不与参考/搜索阶段复用。
    其他 mode 必须提供 N、M、Delta_i。Delta_i 可为完整一维序列、规范
    motif-key 字典（缺项为0），或 hypergraph_deltai_cal 的 by_r 输出。
    序列按 r=2..max_r，其内按 _func_hm_catalog 的 key 字典序排列；默认
    max_r=3/max_v=6 为1496维。当前后端支持 max_r=2或3、2<=max_v<=6，
    其他范围明确报错跳过，不会悄悄漏算。重复超边允许，边内节点不能重复。

    节点编号0..N-1，超边大小2..max_v。null_num 默认100，保持最终图的
    节点度与逐边大小；null 不额外限制连通，避免改变 configuration 空间。
    burn_in 默认 max(100,1000*M)，为经验值，不是严格混合保证。r=2精确；
    r=3在M<=1536或组合数<=exact_limit时精确；适用的大边数网络采用按两侧
    重叠类型分层的 wedge 统计（小层枚举），其他情况复用原 wedge 抽样。
    大边数后端适用于M>1536、M>=N-1且NM<=20000000；其零模型在超边侧
    做等边际Curveball交换，burn_in含义仍为交换次数，默认值不变。
    搜索通过整图参数探索和独立长链复核优化按阶Delta分布及坐标误差。
    exact_limit=0可显式让最终统计使用抽样。搜索按阶数平衡并强调非零目标幅度。
    max_seconds/max_iter 只限制搜索，不截断最终 null 计算。tolerance 只
    用于搜索早停，绝不是保存条件；即使 Delta 不达标，连通候选仍会保存。

    返回值与pkl完全相同，只有以下五项（前四项均为上述同序一维数组）：
    input_delta_i：输入目标；delta_i：最终 (n-mu)/(n+mu+4)；
    hypergraph_hypermotifs_count：最终图计数 n；
    null_model_hypermotifs_count：最终 null_num 个 null 的计数均值 mu；
    hypergraph：tuple节点列表组成的超边列表。不保存误差、状态或其他元数据。
    失败 print 原因并返回 None，不生成pkl。save_path传 result_fig 根目录。
    Windows禁止文件名星号，故文件名用 X.N_M_mode_max_r_max_v.pkl，X取
    目录中已有最大编号+1，不填空缺。相同参数再次调用也会另存一个新编号。
    output_folder=None 时仍保存到 save_path 下的9号文件夹；指定时直接使用
    该完整目录路径，供现实网络批量入口复用，不会先在9号文件夹生成副本。
    '''
    progress = None
    try:
        for value, name, minimum in (
            (max_r, 'max_r', 2), (max_v, 'max_v', 2), (null_num, 'null_num', 2),
            (max_iter, 'max_iter', 0), (search_null_num, 'search_null_num', 2),
            (r3_sample_num, 'r3_sample_num', 2), (exact_limit, 'exact_limit', 0),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if max_r > 3 or max_v > 6:
            raise ValueError('The current motif backend supports max_r=2/3 and max_v=2..6 only.')
        if not isinstance(mode, str) or not mode.strip() or re.search(r'[<>:"/\\|?*\x00-\x1f]', mode):
            raise ValueError('mode must be a nonempty, filename-safe string.')
        mode = mode.strip().lower()
        if not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Invalid tolerance or search time budget.')
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer or None.')
        if burn_in is not None and (isinstance(burn_in, (bool, np.bool_))
                or not isinstance(burn_in, (int, np.integer)) or burn_in < 1):
            raise ValueError('burn_in must be a positive integer or None.')
        rng = np.random.default_rng(seed)
        if mode != 'random' and (N is None or M is None or Delta_i is None):
            raise ValueError('Non-random mode requires N, M and Delta_i.')
        for value, name, minimum in ((N, 'N', 2), (M, 'M', 1)):
            if value is not None and (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if N is None:
            upper = min(10, 1 + M * (max_v - 1)) if M is not None else 10
            N = int(rng.integers(2, upper + 1))
        if M is None:
            minimum = math.ceil((N - 1) / (min(N, max_v) - 1))
            lower = max(minimum, math.ceil(0.4 * N))
            upper = max(lower, math.ceil(1.2 * N))
            M = int(rng.integers(lower, upper + 1))
        N, M = int(N), int(M)
        if N > 1 + M * (min(N, max_v) - 1):
            raise ValueError('Connectedness is impossible: N > 1 + M*(min(N,max_v)-1).')
        burn = max(100, 1000 * M) if burn_in is None else int(burn_in)
        started = time.time()
        progress = tqdm(total = 5, desc = f'Generation N={N}, M={M}', unit = 'stage',
                        dynamic_ncols = True, disable = not verbose)
        keys = _func_hm_generation_keys(int(max_r), int(max_v))
        progress.update(1)
        if Delta_i is None:
            from delta_reconstruction import seed_graph
            # Randomize clustering as well as size; retain unclustered controls.
            reference_rng = np.random.default_rng(_func_hm_seed('generation', 'reference', seed))
            locality = float(reference_rng.choice([0.0, 0.7, 0.95]))
            reference = seed_graph(N, M, max_v, reference_rng, locality=locality)
            reference_counts = _func_hm_generation_counts(
                reference, keys, max_r, max_v, _func_hm_seed('generation', 'reference-count', seed),
                r3_sample_num, exact_limit,
            )
            reference_mean = _func_hm_generation_null_mean(
                reference, N, keys, max_r, max_v, null_num, burn,
                _func_hm_seed('generation', 'reference-null', seed), r3_sample_num,
                exact_limit, 'Random target nulls', verbose,
            )
            target = _func_hm_generation_delta(reference_counts, reference_mean)
            del reference
        else:
            target = _func_hm_generation_target(Delta_i, keys, max_r)
        progress.update(1)
        initial = _func_hm_generation_initial(N, M, max_v, rng)
        generated = _func_hm_generation_search(
            initial, N, target, keys, max_r, max_v,
            _func_hm_seed('generation', 'search', seed), max_iter, max_seconds,
            tolerance, search_null_num, r3_sample_num, exact_limit, verbose,
        )
        if (len(generated) != M or any(not 2 <= len(edge) <= max_v for edge in generated)
                or not _func_hm_generation_connected(generated, N)):
            raise AssertionError('No valid connected candidate was returned.')
        progress.update(1)
        counts = _func_hm_generation_counts(
            generated, keys, max_r, max_v, _func_hm_seed('generation', 'final-count', seed),
            r3_sample_num, exact_limit,
        )
        null_mean = _func_hm_generation_null_mean(
            generated, N, keys, max_r, max_v, null_num, burn,
            _func_hm_seed('generation', 'final-null', seed), r3_sample_num,
            exact_limit, 'Final hypergraph nulls', verbose,
        )
        delta = _func_hm_generation_delta(counts, null_mean)
        if not np.all(np.isfinite(delta)) or np.any(np.abs(delta) > 1.0):
            raise ArithmeticError('Invalid final Delta values.')
        if verbose:
            for r in range(2, max_r + 1):
                mask = np.asarray([key[0] == r for key in keys])
                error = np.abs(delta[mask] - target[mask])
                print(f'  Final independent nulls: r={r}, MAE={error.mean():.6f}, '
                      f'max_error={error.max():.6f}, within_tolerance={bool(error.max() <= tolerance)}',
                      flush = True)
        progress.update(1)
        result = {
            'input_delta_i': target.copy(), 'delta_i': delta,
            'hypergraph_hypermotifs_count': counts,
            'null_model_hypermotifs_count': null_mean,
            'hypergraph': generated,
        }
        output_folder = (
            os.path.join(os.fspath(save_path), '9.generation_model_N_M_delta_cal')
            if output_folder is None else os.fspath(output_folder)
        )
        with _HM_GENERATION_SAVE_LOCK:
            os.makedirs(output_folder, exist_ok = True)
            numbers = [int(match.group(1)) for name in os.listdir(output_folder)
                       if name.endswith('.pkl') and (match := re.match(r'^(\d+)\.', name))]
            number = max(numbers, default = 0) + 1
            output_path = os.path.join(output_folder, f'{number}.{N}_{M}_{mode}_{max_r}_{max_v}.pkl')
            _func_hm_atomic_pickle(result, output_path)
        progress.update(1)
        progress.set_postfix_str(_func_hm_progress_text(started, 5, 5, 'saved'))
        print(f'生成完成，结果已保存：{output_path}')
        return result
    except Exception as error:
        print(f'生成失败，已跳过（N={N}, M={M}, mode={mode}）：{type(error).__name__}: {error}')
        return None
    finally:
        if progress is not None:
            progress.close()


def hypergraph_generation_model_N_M_delta_from_real_data(
        data_path, result_path, save_path = None, max_node_lim = 1000,
        max_r = 3, max_v = 6, null_num = 100, *, seed = None,
        tolerance = 0.05, max_iter = 5000000, max_seconds = 3600.0,
        search_null_num = 10, burn_in = None, r3_sample_num = 200000,
        exact_limit = 20000, verbose = True):
    '''读取现实网络 N/M/Delta，逐个调用现有生成器，不重新生成目标 Delta。

    data_path：包含 link.csv、description.txt 数据集目录的 data 根目录。
    result_path：已有结果根目录，从4.hypergraph_deltai_cal读取
    {数据集目录名}_deltai.pkl。save_path：输出根目录，None时用result_path。
    max_node_lim：只处理 N < max_node_lim 的现实网络，等于上限也跳过；
    N 保留原数据集全部节点（含孤立节点），先依据description筛选，再从
    link.csv核实。M 只计大小2..max_v的超边，与本生成器的边大小范围一致；
    不截断大超边、不去除重复超边身份，也不擅自缩小 N 来满足连通条件。

    目标按规范 motif key 从4号结果中选取 r=2..max_r、边大小<=max_v 的
    完整子向量。既有目标不改写；其 null 若基于全图边际，仍保留原始目标，
    而重构图的 null 基于重构图自身边际，二者不保证能精确匹配。
    检查结果完成标记、范围、epsilon=4及已有数据指纹；缺失、不匹配、
    N/M无法支持连通图或其他单数据集失败时打印原因并继续处理后续网络。

    其他参数含义与 hypergraph_generation_model_N_M_delta 相同，mode固定
    为'real'，N/M/Delta_i自动读取。指定seed时按数据集名派生独立随机种子。
    输出直接写入10.generation_model_N_M_delta_real_data_cal，命名仍为
    X.N_M_real_max_r_max_v.pkl，X从该目录最大编号+1开始，不填空缺。
    每份pkl保持原生成器的五个字段，即使Delta误差未达标也保存连通结果。
    不保存额外清单或元数据；控制台打印来源数据集名，返回成功数据集名称
    列表，避免在批处理内累积所有生成图。重复调用会追加新编号，不去重。
    '''
    completed = []
    try:
        if (isinstance(max_node_lim, (bool, np.bool_))
                or not isinstance(max_node_lim, (int, np.integer)) or max_node_lim < 1):
            raise ValueError('max_node_lim must be a positive integer; the filter is N < max_node_lim.')
        for value, name, lower, upper in ((max_r, 'max_r', 2, 3), (max_v, 'max_v', 2, 6)):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or not lower <= value <= upper):
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}].')
        if seed is not None and (isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError('seed must be a nonnegative integer or None.')
        data_path, result_path = os.fspath(data_path), os.fspath(result_path)
        save_path = result_path if save_path is None else os.fspath(save_path)
        delta_folder = os.path.join(result_path, _HM_RESULT_FOLDER['deltai'])
        output_folder = os.path.join(save_path, '10.generation_model_N_M_delta_real_data_cal')
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f'Data folder not found: {data_path}')
        if not os.path.isdir(delta_folder):
            raise FileNotFoundError(f'Delta result folder not found: {delta_folder}')
    except (TypeError, ValueError, OSError) as error:
        print(f'现实网络重构失败，已跳过：{type(error).__name__}: {error}')
        return completed

    datasets = []
    for dataset_name, dataset_path in _func_hm_dataset_list(data_path):
        try:
            metadata = _func_hm_read_description(dataset_path, dataset_name)
            if metadata['dataset_class'] == 'real':
                datasets.append((dataset_name, dataset_path, metadata))
        except (TypeError, ValueError, OSError) as error:
            print(f'跳过 {dataset_name}：无法读取数据集描述，{error}')

    keys = None
    started = time.time()
    with tqdm(total = len(datasets), desc = 'Real hypergraph reconstruction',
              unit = 'network', dynamic_ncols = True, disable = not verbose) as progress:
        for index, (dataset_name, dataset_path, description) in enumerate(datasets, start = 1):
            try:
                if description['node_num'] >= max_node_lim:
                    print(f'跳过 {dataset_name}：N={description["node_num"]} >= max_node_lim={max_node_lim}')
                    continue
                delta_path = os.path.join(delta_folder, f'{dataset_name}_deltai.pkl')
                if not os.path.isfile(delta_path):
                    raise FileNotFoundError(f'Missing Delta result: {delta_path}')
                edges, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
                node_num = metadata['node_num']
                if node_num >= max_node_lim:
                    print(f'跳过 {dataset_name}：读取link.csv后 N={node_num} >= max_node_lim={max_node_lim}')
                    continue
                total_edge_num = len(edges)
                edge_num = sum(2 <= len(edge) <= max_v for edge in edges)
                del edges
                if node_num < 2 or edge_num < 1:
                    raise ValueError('No connected size-2..max_v reconstruction is possible with this N/M.')
                if node_num > 1 + edge_num * (min(node_num, max_v) - 1):
                    raise ValueError('N exceeds the connectedness capacity of the in-range hyperedges.')

                with open(delta_path, 'rb') as file:
                    delta_data = pkl.load(file)
                if not isinstance(delta_data, dict) or delta_data.get('complete') is not True:
                    raise ValueError('Delta result is not marked complete.')
                if delta_data.get('dataset_name', dataset_name) != dataset_name:
                    raise ValueError('Delta result belongs to a different dataset.')
                settings = delta_data.get('settings', {})
                if float(delta_data.get('epsilon', settings.get('epsilon', 4.0))) != 4.0:
                    raise ValueError('Delta epsilon differs from the generator epsilon=4.')
                if int(settings.get('maximum_cardinality', max_v)) < max_v:
                    raise ValueError('Stored Delta results cover a smaller cardinality range.')
                fingerprint = delta_data.get('source_fingerprint')
                if fingerprint and fingerprint != _func_hm_source_fingerprint(dataset_path):
                    raise ValueError('Data changed since Delta was calculated; recompute the source Delta first.')
                stored_metadata = delta_data.get('metadata', {})
                if ('node_num' in stored_metadata and int(stored_metadata['node_num']) != node_num
                        or 'edge_num' in stored_metadata and int(stored_metadata['edge_num']) != total_edge_num):
                    raise ValueError('Delta metadata N/M does not match the source data.')
                if keys is None:
                    keys = _func_hm_generation_keys(int(max_r), int(max_v))
                target = {}
                for r in range(2, max_r + 1):
                    by_r = delta_data.get('by_r', {}).get(r)
                    if not isinstance(by_r, dict) or not isinstance(by_r.get('deltai'), dict):
                        raise ValueError(f'Missing canonical Delta dictionary for r={r}.')
                    required = [key for key in keys if key[0] == r]
                    if any(key not in by_r['deltai'] for key in required):
                        raise ValueError(f'Incomplete Delta catalogue for r={r}; missing entries cannot be assumed zero.')
                    target.update({key: by_r['deltai'][key] for key in required})
                # 通过 key 对齐，绝不依赖保存字典的偶然遍历顺序。
                target = _func_hm_generation_target(target, keys, max_r)
                del delta_data
                print(f'现实网络重构：{dataset_name}，N={node_num}，M={edge_num}，'
                      f'范围外超边数={total_edge_num - edge_num}')
                dataset_seed = None if seed is None else _func_hm_seed(dataset_name, 'real-reconstruction', seed)
                generated = hypergraph_generation_model_N_M_delta(
                    save_path, mode = 'real', N = node_num, M = edge_num, Delta_i = target,
                    max_r = max_r, max_v = max_v, null_num = null_num, seed = dataset_seed,
                    tolerance = tolerance, max_iter = max_iter, max_seconds = max_seconds,
                    search_null_num = search_null_num, burn_in = burn_in,
                    r3_sample_num = r3_sample_num, exact_limit = exact_limit, verbose = verbose,
                    output_folder = output_folder,
                )
                if generated is not None:
                    completed.append(dataset_name)
                    del generated
                else:
                    print(f'跳过 {dataset_name}：生成器未返回成功保存的连通超图。')
            except Exception as error:
                print(f'跳过 {dataset_name}：{type(error).__name__}: {error}')
            finally:
                progress.update(1)
                if verbose:
                    progress.set_postfix_str(_func_hm_progress_text(
                        started, index, len(datasets), dataset_name
                    ), refresh = False)
    print(f'现实网络重构完成：成功保存 {len(completed)} 个，共扫描 {len(datasets)} 个现实数据集。')
    return completed


# ------------------------现实网络重构前后分布比较------------------------

def _func_hm_comparison_file_hash(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _func_hm_comparison_matches(data_path, generation_folder):
    '''按全部节点 N、范围内超边 M、max_v 建索引，不使用近似匹配。'''
    records = []
    for name in sorted(os.listdir(generation_folder), key = _func_hm_dataset_sort_key):
        if not name.endswith('.pkl'):
            continue
        match = re.fullmatch(r'(\d+)\.(\d+)_(\d+)_(.+)_(\d+)_(\d+)\.pkl', name)
        if match is None:
            print(f'跳过 {name}：无法解析重构结果文件名。')
            continue
        number, node_num, edge_num, mode, max_r, max_v = match.groups()
        node_num, edge_num, max_r, max_v = map(int, (node_num, edge_num, max_r, max_v))
        if mode != 'real' or node_num < 2 or edge_num < 1 or max_r not in (2, 3) or not 2 <= max_v <= 6:
            print(f'跳过 {name}：不是支持范围内的现实网络重构结果。')
            continue
        records.append({'file_name': name, 'N': node_num, 'M': edge_num,
                        'max_r': max_r, 'max_v': max_v})
    requested = {(record['N'], record['max_v']) for record in records}
    candidates = defaultdict(list)
    datasets = _func_hm_dataset_list(data_path)
    started = time.time()
    with tqdm(datasets, desc = 'Match real N/M', unit = 'network', dynamic_ncols = True) as bar:
        for index, (dataset_name, dataset_path) in enumerate(bar, start = 1):
            metadata = _func_hm_read_description(dataset_path, dataset_name)
            if metadata['dataset_class'] == 'real':
                declared_n = metadata['node_num']
                if declared_n == 0 or any(n == declared_n for n, v in requested):
                    # 索引读取失败时向上报错，不能用不完整候选集宣称唯一匹配。
                    edges, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
                    n = metadata['node_num']
                    for max_v in sorted({v for size, v in requested if size == n}):
                        m = sum(2 <= len(edge) <= max_v for edge in edges)
                        candidates[(n, m, max_v)].append((dataset_name, dataset_path))
                    del edges
            bar.set_postfix_str(_func_hm_progress_text(started, index, len(datasets), dataset_name))
    groups = {}
    for record in records:
        matches = candidates[(record['N'], record['M'], record['max_v'])]
        if len(matches) != 1:
            names = ', '.join(name for name, path in matches) or 'none'
            print(f'跳过 {record["file_name"]}：点边数量精确匹配到 {len(matches)} 个数据集：{names}')
            continue
        dataset_name, dataset_path = matches[0]
        group = groups.setdefault(dataset_name, {'path': dataset_path, 'records': []})
        group['records'].append(record)
    return groups


def _func_hm_comparison_distribution(edges, node_num, metric):
    if metric == 'degree':
        values = np.zeros(node_num, dtype = np.int64)
        for edge in edges:
            for node in edge:
                values[node] += 1
    else:
        values = np.asarray([len(edge) for edge in edges], dtype = np.int64)
    # Counter 保留零度节点，也保留重复超边的独立贡献。
    return Counter(values.tolist())


def _func_hm_comparison_ci(counts, keys, max_r):
    counts = np.asarray(counts, dtype = float)
    if counts.shape != (len(keys),) or not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError('Motif counts have invalid shape, nonfinite or negative values.')
    result = {}
    for r in range(2, max_r + 1):
        indices = [index for index, key in enumerate(keys) if key[0] == r]
        subset = counts[indices]
        total = float(subset.sum())
        if not np.isfinite(total):
            raise ValueError('Motif count sum is not finite.')
        result[r] = {
            'x': np.arange(1, len(indices) + 1, dtype = np.int64),
            'motif_keys': tuple(keys[index] for index in indices),
            'counts': subset, 'total_count': total,
            'y': subset / total if total > 0 else np.zeros(len(indices)),
            'has_motifs': total > 0,
        }
    return result


def _func_hm_real_generation_comparasion(data_path, save_path, metric):
    folder_number = {'degree': 11, 'cardinality': 12, 'ci': 13}[metric]
    generation_folder = os.path.join(save_path, '10.generation_model_N_M_delta_real_data_cal')
    output_folder = os.path.join(save_path, f'{folder_number}.real_generation_{metric}_comparasion_cal')
    os.makedirs(output_folder, exist_ok = True)
    if not os.path.isdir(data_path) or not os.path.isdir(generation_folder):
        print(f'{metric}比较已跳过：data或10号结果文件夹不存在。')
        return []
    try:
        groups = _func_hm_comparison_matches(data_path, generation_folder)
    except Exception as error:
        print(f'{metric}比较已停止：无法构建完整的点边匹配索引，{error}')
        return []
    completed, started = [], time.time()
    with tqdm(groups.items(), total = len(groups), desc = f'Real comparison {metric}',
              unit = 'network', dynamic_ncols = True) as outer:
        for index, (dataset_name, group) in enumerate(outer, start = 1):
            try:
                dataset_path, records = group['path'], group['records']
                fingerprint = _func_hm_source_fingerprint(dataset_path)
                source_count_path = os.path.join(save_path, _HM_RESULT_FOLDER['motif'],
                                                f'{dataset_name}_hypermotifs.pkl')
                dependencies = {
                    record['file_name']: _func_hm_comparison_file_hash(
                        os.path.join(generation_folder, record['file_name'])) for record in records
                }
                if metric == 'ci' and os.path.isfile(source_count_path):
                    dependencies['original_motifs'] = _func_hm_comparison_file_hash(source_count_path)
                signature = _func_hm_signature((metric, fingerprint, dependencies, 'comparison-v1'))
                output_path = os.path.join(output_folder, f'{dataset_name}_{metric}.pkl')
                if os.path.isfile(output_path):
                    try:
                        with open(output_path, 'rb') as file:
                            previous = pkl.load(file)
                        if (previous.get('schema_version') == 'real-comparison-v1'
                                and previous.get('complete') is True
                                and previous.get('settings_signature') == signature):
                            completed.append(output_path)
                            print(f'跳过已完成结果：{dataset_name}_{metric}')
                            continue
                    except Exception:
                        pass
                original_edges, metadata = _func_hm_read_hypergraph(dataset_path, dataset_name)
                source_counts = None
                if metric == 'ci' and os.path.isfile(source_count_path):
                    try:
                        with open(source_count_path, 'rb') as file:
                            source_counts = pkl.load(file)
                        if (source_counts.get('complete') is not True
                                or source_counts.get('source_fingerprint') != fingerprint
                                or source_counts.get('dataset_name', dataset_name) != dataset_name):
                            source_counts = None
                    except Exception:
                        source_counts = None
                comparisons, failures, original_ci_cache = [], [], {}
                local_started = time.time()
                with tqdm(records, desc = dataset_name, unit = 'reconstruction',
                          dynamic_ncols = True, leave = False) as inner:
                    for item_index, record in enumerate(inner, start = 1):
                        name = record['file_name']
                        try:
                            n, m, max_r, max_v = (record[key] for key in ('N', 'M', 'max_r', 'max_v'))
                            before_edges = [edge for edge in original_edges if 2 <= len(edge) <= max_v]
                            if metadata['node_num'] != n or len(before_edges) != m:
                                raise ValueError('Original N/M no longer exactly matches the generation result.')
                            with open(os.path.join(generation_folder, name), 'rb') as file:
                                generated = pkl.load(file)
                            required = {'input_delta_i', 'delta_i', 'hypergraph_hypermotifs_count',
                                        'null_model_hypermotifs_count', 'hypergraph'}
                            if not isinstance(generated, dict) or not required <= set(generated):
                                raise ValueError('Missing one or more of the five generation result fields.')
                            after_edges = generated['hypergraph']
                            if (len(after_edges) != m or any(not 2 <= len(edge) <= max_v for edge in after_edges)
                                    or any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_))
                                           for edge in after_edges for node in edge)
                                    or not _func_hm_generation_connected(after_edges, n)):
                                raise ValueError('Generated hypergraph does not satisfy the filename N/M and connectedness.')
                            comparison = dict(record)
                            comparison['excluded_original_edge_num'] = len(original_edges) - m
                            if metric in ('degree', 'cardinality'):
                                before = _func_hm_comparison_distribution(before_edges, n, metric)
                                after = _func_hm_comparison_distribution(after_edges, n, metric)
                                x = np.asarray(sorted(set(before) | set(after)), dtype = np.int64)
                                comparison.update({
                                    'x': x,
                                    'before_y': np.asarray([before[int(value)] for value in x], dtype = np.int64),
                                    'after_y': np.asarray([after[int(value)] for value in x], dtype = np.int64),
                                    'x_label': f'{metric.capitalize()} value',
                                    'y_label': 'Node count' if metric == 'degree' else 'Hyperedge count',
                                })
                            else:
                                keys = _func_hm_generation_keys(max_r, max_v)
                                cache_key = (max_r, max_v)
                                if cache_key not in original_ci_cache:
                                    r_values = tuple(range(2, max_r + 1))
                                    reusable = (source_counts is not None
                                                and source_counts.get('settings', {}).get('maximum_cardinality') == max_v
                                                and all(r in source_counts.get('by_r', {}) for r in r_values))
                                    if reusable:
                                        by_r = source_counts['by_r']
                                        count_source = source_count_path
                                    else:
                                        by_r = alg_hypermotifs_mining(
                                            before_edges, r_values = r_values, t = max_v,
                                            r3_sample_num = 200000,
                                            seed = _func_hm_seed(dataset_name, 'comparison-original', _HM_DEFAULT_SEED),
                                            progress_desc = f'{dataset_name} original motifs',
                                        )
                                        count_source = 'recomputed: r=2 exact; r=3 wedge estimate, sample_num=200000'
                                    counts = [by_r[key[0]]['counts'].get(key, 0.0) for key in keys]
                                    original_ci_cache[cache_key] = (
                                        _func_hm_comparison_ci(counts, keys, max_r), count_source,
                                        {r: by_r[r].get('method') for r in r_values},
                                    )
                                before, count_source, methods = original_ci_cache[cache_key]
                                after = _func_hm_comparison_ci(generated['hypergraph_hypermotifs_count'], keys, max_r)
                                comparison['by_r'] = {
                                    r: {'x': before[r]['x'], 'motif_keys': before[r]['motif_keys'],
                                        'before_y': before[r]['y'], 'after_y': after[r]['y'],
                                        'before_counts': before[r]['counts'], 'after_counts': after[r]['counts'],
                                        'before_total': before[r]['total_count'], 'after_total': after[r]['total_count'],
                                        'before_has_motifs': before[r]['has_motifs'], 'after_has_motifs': after[r]['has_motifs']}
                                    for r in range(2, max_r + 1)
                                }
                                comparison.update({'normalization': 'within_each_exact_r',
                                                   'original_count_source': count_source,
                                                   'original_count_methods': methods,
                                                   'generated_count_source': name,
                                                   'x_label': 'Hyper-motif ID', 'y_label': 'C_i'})
                            comparisons.append(comparison)
                        except Exception as error:
                            failures.append({'file_name': name, 'reason': str(error)})
                            print(f'跳过 {name}：{metric}比较失败，{error}')
                        finally:
                            inner.set_postfix_str(_func_hm_progress_text(
                                local_started, item_index, len(records), name))
                if comparisons:
                    _func_hm_atomic_pickle({
                        'schema_version': 'real-comparison-v1', 'complete': not failures,
                        'dataset_name': dataset_name, 'metric': metric,
                        'source_fingerprint': fingerprint, 'settings_signature': signature,
                        'input_fingerprints': dependencies,
                        'scope': 'all original nodes; only hyperedges of size 2..max_v; duplicate edges retained',
                        'comparisons': comparisons, 'failures': failures,
                    }, output_path)
                    completed.append(output_path)
            except Exception as error:
                print(f'跳过 {dataset_name}：{metric}比较失败，{error}')
            finally:
                outer.set_postfix_str(_func_hm_progress_text(started, index, len(groups), dataset_name))
    return completed


def real_generation_degree_comparasion(data_path: str, save_path: str):
    '''保存11号文件夹/{原数据集名}_degree.pkl，save_path为result_fig根目录。

    按N和大小2..max_v范围内的M精确唯一匹配10号重构结果；歧义时跳过。
    comparisons每项对应一次重构，x为degree，before_y/after_y为节点数；
    包含零度节点，重复超边按独立身份计度，同一网络多次重构不会覆盖。
    输入哈希不变且结果完整时跳过；返回已完成结果路径列表，不绘图。
    '''
    return _func_hm_real_generation_comparasion(data_path, save_path, 'degree')


def real_generation_cardinality_comparasion(data_path: str, save_path: str):
    '''保存12号文件夹/{原数据集名}_cardinality.pkl。

    匹配、范围及续算规则与degree比较一致。comparisons每项中，x为超边
    cardinality，before_y/after_y为对应超边数量；不同重复超边分别计数。
    '''
    return _func_hm_real_generation_comparasion(data_path, save_path, 'cardinality')


def real_generation_ci_comparasion(data_path: str, save_path: str):
    '''保存13号文件夹/{原数据集名}_ci.pkl，不从Delta反推C_i。

    重构后直接使用10号文件的hypergraph_hypermotifs_count。重构前优先
    复用2号文件夹中指纹和范围匹配的原图计数，否则复用现有挖掘器计算；
    r=2精确、r=3默认20万次抽样估计，不重新生成null或改动其他结果文件。
    comparisons每项的by_r分别归一化：C_i=count_i/sum(counts_r)。
    每个r中x为从1开始的motif ID，motif_keys保留无标签结构对应关系，
    before_y/after_y可直接绘图。总计数为零时y全零且has_motifs=False。
    '''
    return _func_hm_real_generation_comparasion(data_path, save_path, 'ci')


# ------------------------SI：逐网络 Delta 对比数据------------------------

_HM_SI_DELTA_FOLDER = '15.hypergraph_generation_deltai_comparasion_cal'
_HM_SI_DELTA_VERSION = 'si-delta-v1'


def _func_hm_si_delta_record(source_path, source_folder):
    '''纯读取、校验和拆分；返回值不含运行时间，便于独立验证复现。'''
    name = os.path.basename(source_path)
    match = re.fullmatch(r'(\d+)\.(\d+)_(\d+)_(random|real)_(\d+)_(\d+)\.pkl', name)
    if match is None:
        raise ValueError(f'Invalid reconstruction filename: {name}')
    number, n, m, mode, max_r, max_v = match.groups()
    n, m, max_r, max_v = map(int, (n, m, max_r, max_v))
    if n < 2 or m < 1 or max_r not in (2, 3) or not 2 <= max_v <= 6:
        raise ValueError(f'Unsupported reconstruction parameters: {name}')
    expected = 'random' if source_folder.startswith('9.') else 'real'
    if mode != expected:
        raise ValueError(f'{name}: mode does not match source folder.')
    with open(source_path, 'rb') as file:
        data = pkl.load(file)
    keys = _func_hm_generation_keys(max_r, max_v)
    before = _func_hm_generation_target(data['input_delta_i'], keys, max_r)
    after = _func_hm_generation_target(data['delta_i'], keys, max_r)
    counts = np.asarray(data['hypergraph_hypermotifs_count'], dtype = float)
    null_mean = np.asarray(data['null_model_hypermotifs_count'], dtype = float)
    for values in (counts, null_mean):
        if values.shape != before.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f'{name}: invalid motif counts.')
    if not np.allclose(after, _func_hm_generation_delta(counts, null_mean), rtol = 0, atol = 1e-12):
        raise ValueError(f'{name}: Delta disagrees with the stored counts.')
    edges = data['hypergraph']
    if (len(edges) != m or any(not 2 <= len(edge) <= max_v for edge in edges)
            or any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_))
                   for edge in edges for node in edge)
            or not _func_hm_generation_connected(edges, n)):
        raise ValueError(f'{name}: invalid connected hypergraph or N/M mismatch.')
    by_r = {}
    for r in (2, 3):
        indices = [index for index, key in enumerate(keys) if key[0] == r]
        by_r[r] = {
            'available': bool(indices),
            'motif_keys': tuple(keys[index] for index in indices),
            'x': np.arange(1, len(indices) + 1, dtype = np.int64),
            'before_y': before[indices].copy(), 'after_y': after[indices].copy(),
        }
    return {
        'schema_version': _HM_SI_DELTA_VERSION, 'complete': True,
        'source_folder': source_folder, 'source_file': name,
        'source_fingerprint': _func_hm_comparison_file_hash(source_path),
        'network_class': 'artificial' if mode == 'random' else 'real',
        'generation_number': int(number), 'N': n, 'M': m, 'mode': mode,
        'max_r': max_r, 'max_v': max_v, 'by_r': by_r,
        'ordering': 'canonical motif order within each exact r; zeros retained',
    }


def hypergraph_generation_deltai_comparasion(save_path: str):
    '''读取9/10号结果，向15号写出逐网络r=2/3输入与重构Delta，不重新挖掘。

    save_path为result_fig根目录；结果名为{原文件stem}_deltai.pkl。
    哈希和版本相同且已完成时跳过。失败不生成完成结果，返回files/skipped/errors。
    '''
    output_folder = os.path.join(save_path, _HM_SI_DELTA_FOLDER)
    os.makedirs(output_folder, exist_ok = True)
    report = {'files': [], 'skipped': [], 'errors': []}
    sources = []
    for folder in ('9.generation_model_N_M_delta_cal', '10.generation_model_N_M_delta_real_data_cal'):
        path = os.path.join(save_path, folder)
        if not os.path.isdir(path):
            report['errors'].append({'file': folder, 'reason': 'Source directory not found.'})
            print(f'SI Delta: missing folder {folder}')
            continue
        sources.extend((folder, name) for name in sorted(os.listdir(path), key = _func_hm_dataset_sort_key)
                       if name.endswith('.pkl') and not name.startswith('_'))
    started = time.time()
    with tqdm(sources, desc = 'SI Delta total', unit = 'network', dynamic_ncols = True) as bar:
        for index, (folder, name) in enumerate(bar, start = 1):
            try:
                source_path = os.path.join(save_path, folder, name)
                output_path = os.path.join(output_folder, f'{os.path.splitext(name)[0]}_deltai.pkl')
                fingerprint = _func_hm_comparison_file_hash(source_path)
                if os.path.isfile(output_path):
                    try:
                        with open(output_path, 'rb') as file:
                            cached = pkl.load(file)
                        if (cached.get('schema_version') == _HM_SI_DELTA_VERSION
                                and cached.get('complete') is True
                                and cached.get('source_fingerprint') == fingerprint
                                and cached.get('source_folder') == folder and cached.get('source_file') == name):
                            report['files'].append(output_path)
                            report['skipped'].append(output_path)
                            continue
                    except Exception:
                        pass
                record = _func_hm_si_delta_record(source_path, folder)
                if record['source_fingerprint'] != fingerprint:
                    raise ValueError('Source changed during processing; rerun with stable input.')
                _func_hm_atomic_pickle(record, output_path)
                report['files'].append(output_path)
            except Exception as error:
                report['errors'].append({'file': name, 'reason': str(error)})
                print(f'SI Delta skipped {name}: {error}')
            finally:
                bar.set_postfix_str(_func_hm_progress_text(started, index, len(sources), name))
    print(f'SI Delta: {len(report["files"])} complete, {len(report["skipped"])} cached, '
          f'{len(report["errors"])} errors.')
    return report


# ------------------------18：重构精度与可复核原始数据------------------------

_HM_RECONSTRUCTION_METRICS = (
    'MAE2', 'MAE3', 'Active MAE2', 'Active MAE3', 'W1_2', 'W1_3', 'E_inf',
)


def _func_hm_reconstruction_vector(values, keys, name):
    '''完整规范向量；缺失的 motif 不能被补成零。'''
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate canonical motif keys.')
    if isinstance(values, dict):
        if set(values) != set(keys):
            raise ValueError(f'{name}: missing or unknown canonical motif keys.')
        values = [values[key] for key in keys]
    result = np.asarray(values, dtype = float)
    if result.shape != (len(keys),) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name}: incomplete, nonfinite or incorrectly shaped vector.')
    return result


def _func_hm_reconstruction_metrics(target, reconstructed, keys, active_threshold):
    '''按同一 motif ID 计算坐标误差，仅 Wasserstein 比较独立排序的数值。'''
    if not np.isfinite(active_threshold) or not 0 <= active_threshold <= 1:
        raise ValueError('active_threshold must be finite and within [0, 1].')
    target = _func_hm_reconstruction_vector(target, keys, 'target')
    reconstructed = _func_hm_reconstruction_vector(reconstructed, keys, 'reconstructed')
    if any(np.any(np.abs(a) > 1) for a in (target, reconstructed)):
        raise ValueError('Abundances must lie within [-1, 1].')
    if set(key[0] for key in keys) != {2, 3}:
        raise ValueError('Both motif orders r=2 and r=3 are required.')
    error = np.abs(target - reconstructed)
    values, active_counts = {}, {}
    for r in (2, 3):
        mask = np.asarray([key[0] == r for key in keys])
        tar, rec, err = target[mask], reconstructed[mask], error[mask]
        active = np.abs(tar) >= active_threshold
        active_counts[r] = int(active.sum())
        values[f'MAE{r}'] = float(err.mean())
        values[f'Active MAE{r}'] = float(err[active].mean()) if active.any() else None
        values[f'W1_{r}'] = float(np.abs(np.sort(tar) - np.sort(rec)).mean())
    values['E_inf'] = float(error.max())
    return values, active_counts


def _func_hm_reconstruction_summary(records):
    result = {name: {} for name in ('Median', 'IQR', 'Maximum', 'Valid N')}
    for metric in _HM_RECONSTRUCTION_METRICS:
        values = np.asarray([row['metrics'][metric] for row in records
                             if row['metrics'][metric] is not None], dtype = float)
        result['Valid N'][metric] = int(values.size)
        if not values.size:
            for name in ('Median', 'IQR', 'Maximum'):
                result[name][metric] = None
            continue
        # NumPy's default linear interpolation matches Excel PERCENTILE.INC.
        q1, q3 = np.quantile(values, [0.25, 0.75], method = 'linear')
        result['Median'][metric] = float(np.median(values))
        result['IQR'][metric] = float(q3 - q1)
        result['Maximum'][metric] = float(values.max())
    return result


# The workbook builder is kept here so the public Python entry point owns the
# complete export. Node and @oai/artifact-tool are supplied by the desktop runtime.
# For another installation, set HM_ARTIFACT_NODE and HM_ARTIFACT_MODULES.
_HM_RECONSTRUCTION_WORKBOOK_JS = r'''
import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const data = JSON.parse(await fs.readFile(process.argv[1], 'utf8'));
const wb = Workbook.create();
const metricNames = ['MAE2','MAE3','Active MAE2','Active MAE3','W1_2','W1_3','E_inf'];
const metricHeaders = ['MAE²','MAE³','Active MAE²','Active MAE³','W₁²','W₁³','E∞'];
function style(sh, lastRow, lastCol, headerRow) {
  sh.showGridLines = false;
  const area = sh.getRangeByIndexes(0,0,lastRow,lastCol);
  area.format.font.name = 'Arial'; area.format.font.size = 10;
  area.format.rowHeight = 15; area.format.verticalAlignment = 'center';
  area.format.columnWidth = 15;
  const header = sh.getRangeByIndexes(headerRow-1,0,1,lastCol);
  header.format.fill = '#33475B'; header.format.font.color = '#FFFFFF';
  header.format.font.bold = true; header.format.wrapText = true;
  header.format.horizontalAlignment = 'center'; header.format.rowHeight = 30;
  sh.freezePanes.freezeRows(headerRow); sh.freezePanes.freezeColumns(1);
}
for (const [group, name] of [['empirical','Empirical'],['artificial','Artificial']]) {
  const sh = wb.worksheets.add(name), rows = data.groups[group];
  const headers = ['Network ID',...metricHeaders,'Name','Category','N','M',
    'Generation ID','Active classes r=2','Active classes r=3','Dataset folder',
    'Source file','SHA-256'];
  const cells = rows.map(x => [x.id,...metricNames.map(k => x.metrics[k] ?? 'N/A'),
    x.name,x.category,x.N,x.M,x.generation,x.active_counts['2'],x.active_counts['3'],
    x.dataset,x.source_file,x.sha256]);
  const headerRow = 5, first = 6, last = 5 + cells.length;
  sh.getRange('A2').values = [[name + ' reconstruction accuracy']];
  sh.getRange('A3').values = [[`Final independent evaluation; active |target| >= ${data.threshold}; J2=50, J3=1446.`]];
  sh.getRange('A5:R5').values = [headers];
  sh.getRangeByIndexes(5,0,cells.length,18).values = cells;
  sh.tables.add(`A5:R${last}`,true,`${name}Networks`).showFilterButton = true;
  for (let j=0;j<4;j++) {
    const stat = ['Median','IQR','Maximum','Valid N'][j];
    sh.getRangeByIndexes(last+1+j,0,1,8).values = [[stat,
      ...metricNames.map(k => data.summaries[group][stat][k] ?? 'N/A')]];
  }
  const noteRow = last+8;
  const notes = [
    'All errors are dimensionless. Superscripts 2 and 3 denote motif order.',
    'MAE uses matching motif IDs, retaining all zeros. Active MAE uses only |target| >= the stated threshold.',
    'W1 is the mean absolute difference of independently sorted target and reconstructed values.',
    'E∞ is the largest error across both orders. N/A means no active motif; it is excluded from summaries.',
    'IQR = Q75 - Q25 using linear interpolation. Valid N is the number of contributing networks.',
    'N and M describe the reconstructed graph; empirical M includes edge cardinalities 2–6 only.',
    'Raw data preserve canonical coordinates; sorted-rank columns are separate order statistics, not motif identities.',
    'Values are calculated in Python without prior rounding. Rerun the calculation after changing source files.'
  ];
  for (let i=0;i<notes.length;i++) sh.getRange(`A${noteRow+i}`).values = [[notes[i]]];
  style(sh,noteRow+notes.length,18,headerRow);
  sh.getRange('A2').format.font.size = 14; sh.getRange('A2').format.font.bold = true;
  sh.getRange('A3').format.font.italic = true;
  sh.getRange(`B${first}:H${last+4}`).setNumberFormat('0.000000');
  sh.getRange(`B${last+5}:H${last+5}`).setNumberFormat('0');
  sh.getRange(`A${last+2}:H${last+5}`).format.font.bold = true;
  sh.getRange(`A${last+2}:H${last+5}`).format.fill = '#EDF1F5';
  sh.getRange('I:I').format.columnWidth = 43;
  sh.getRange('J:J').format.columnWidth = 23;
  sh.getRange('P:Q').format.columnWidth = 52;
  sh.getRange('R:R').format.columnWidth = 68;
  sh.getRange(`B${first}:O${last+5}`).format.horizontalAlignment = 'right';
  sh.getRange(`I${first}:J${last}`).format.horizontalAlignment = 'left';
}
const raw = wb.worksheets.add('Raw data');
const headers = ['Network type','Network ID','r','Motif ID','Canonical key',
 'Target abundance','Reconstructed abundance','Final motif count','Final null mean',
 'Absolute error','Active','Sorted rank','Sorted target','Sorted reconstructed','Sorted absolute difference'];
raw.getRange('A1:O1').values = [headers];
for (let offset=0;offset<data.raw.length;offset+=5000) {
  const part = data.raw.slice(offset,offset+5000);
  raw.getRangeByIndexes(offset+1,0,part.length,15).values = part;
}
const rawLast = data.raw.length+1;
raw.tables.add(`A1:O${rawLast}`,true,'MotifRawData').showFilterButton = true;
style(raw,rawLast,15,1);
raw.getRange('E:E').format.columnWidth = 31;
raw.getRange('F:J').format.columnWidth = 23;
raw.getRange('L:O').format.columnWidth = 23;
raw.getRange(`F2:J${rawLast}`).setNumberFormat('0.000000');
raw.getRange(`M2:O${rawLast}`).setNumberFormat('0.000000');
wb.recalculate();
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(process.argv[2]);
if (process.env.HM_RECONSTRUCTION_PREVIEW) {
  const dir = process.env.HM_RECONSTRUCTION_PREVIEW;
  await fs.mkdir(dir,{recursive:true});
  for (const [sheetName,range] of [['Empirical','A2:H23'],['Artificial','A2:H16'],['Raw data','A1:O9']]) {
    const blob = await wb.render({sheetName,range,scale:1.5,format:'png'});
    await fs.writeFile(`${dir}/${sheetName.replaceAll(' ','_')}.png`,new Uint8Array(await blob.arrayBuffer()));
  }
}
console.log(JSON.stringify({sheets:3,networks:data.groups.empirical.length+data.groups.artificial.length,
 raw_rows:data.raw.length,output:process.argv[2]}));
'''


def hypergraph_reconstruction_sta_cal(data_path, save_path, *, active_threshold=0.1):
    '''从9/10号原始结果生成18.reconstruction_sta_cal/reconstruction_accuracy.xlsx。

    全量校验后再替换工作簿；不重构网络、不改变原始数据，不用图片估算。
    11号记录只用于现实网络身份映射；15号已有记录用于核对规范编号及数值。
    所有输入必须为r=2/3、v=6，保留50/1446个坐标。缺失值不会被补零。
    Excel有Empirical、Artificial、Raw data三页，浮点数不预先舍入；N/A为文本。
    IQR=Q75-Q25，线性插值；统计按网络等权，排除无active motif的Active MAE。
    返回excel_path、groups、summaries、raw_rows，便于复核或排版SI表。
    导出使用Node和@oai/artifact-tool，默认自动定位Codex运行时，也可通过
    HM_ARTIFACT_NODE/HM_ARTIFACT_MODULES指定安装位置。
    '''
    from pathlib import Path
    import subprocess
    root, data_root = Path(save_path).resolve(), Path(data_path).resolve()
    if isinstance(active_threshold, (bool, np.bool_)) or not np.isfinite(active_threshold) or not 0 <= active_threshold <= 1:
        raise ValueError('active_threshold must be a number within [0, 1].')
    keys = _func_hm_generation_keys(3, 6)
    catalog = {r: tuple(k for k in keys if k[0] == r) for r in (2, 3)}
    if (len(catalog[2]), len(catalog[3])) != (50, 1446) or len(set(keys)) != 1496:
        raise ValueError('Expected the complete canonical catalogue J2=50, J3=1446.')
    catalog_path = root / '2.hypermotifs_sta_cal' / 'hypermotif_catalog.pkl'
    if catalog_path.exists():
        with catalog_path.open('rb') as stream:
            stored = pkl.load(stream)
        if not stored.get('complete') or any(tuple(stored['catalog'][r]) != catalog[r] for r in (2, 3)):
            raise ValueError('Stored motif catalogue differs from the current canonical indexing.')
    real_map = {}
    for path in sorted((root / '11.real_generation_degree_comparasion_cal').glob('*.pkl')):
        with path.open('rb') as stream:
            mapping = pkl.load(stream)
        if not mapping.get('complete'):
            raise ValueError(f'Incomplete empirical identity record: {path.name}')
        for item in mapping['comparisons']:
            name = item['file_name']
            if name in real_map:
                raise ValueError(f'Ambiguous empirical identity: {name}')
            real_map[name] = (mapping['dataset_name'], item,
                              mapping['input_fingerprints'][name])
    groups, raw_rows = {'empirical': [], 'artificial': []}, []
    source_checks = []
    for group, folder in [('empirical', '10.generation_model_N_M_delta_real_data_cal'),
                          ('artificial', '9.generation_model_N_M_delta_cal')]:
        paths = sorted((root / folder).glob('*.pkl'), key = lambda p: _func_hm_dataset_sort_key(p.name))
        if not paths:
            raise ValueError(f'No reconstruction records in {folder}.')
        seen_ids = set()
        for path in paths:
            match = re.fullmatch(r'(\d+)\.(\d+)_(\d+)_(real|random)_3_6\.pkl', path.name)
            if match is None or (match[4] == 'real') != (group == 'empirical'):
                raise ValueError(f'Unsupported reconstruction record: {path.name}')
            generation, n, m = map(int, match.groups()[:3])
            sha = _func_hm_comparison_file_hash(str(path))
            with path.open('rb') as stream:
                saved = pkl.load(stream)
            target = _func_hm_reconstruction_vector(saved['input_delta_i'], keys, path.name + ' target')
            final = _func_hm_reconstruction_vector(saved['delta_i'], keys, path.name + ' final')
            counts = _func_hm_reconstruction_vector(saved['hypergraph_hypermotifs_count'], keys, 'counts')
            null = _func_hm_reconstruction_vector(saved['null_model_hypermotifs_count'], keys, 'null mean')
            if np.any(counts < 0) or np.any(null < 0) or not np.allclose(
                    final, _func_hm_generation_delta(counts, null), rtol=0, atol=1e-12):
                raise ValueError(f'{path.name}: final abundance disagrees with its counts.')
            edges = saved['hypergraph']
            if (len(edges) != m or any(not 2 <= len(edge) <= 6 for edge in edges)
                    or any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_))
                           for edge in edges for node in edge)
                    or not _func_hm_generation_connected(edges, n)):
                raise ValueError(f'{path.name}: invalid hypergraph or N/M mismatch.')
            cached_path = root / _HM_SI_DELTA_FOLDER / (path.stem + '_deltai.pkl')
            if cached_path.exists():
                with cached_path.open('rb') as stream:
                    cached = pkl.load(stream)
                if not cached.get('complete') or cached.get('source_fingerprint') != sha:
                    raise ValueError(f'{path.name}: stale or incomplete plotting data.')
                for r, sl in [(2, slice(0, 50)), (3, slice(50, 1496))]:
                    item = cached['by_r'][r]
                    if (tuple(item['motif_keys']) != catalog[r]
                            or not np.array_equal(item['x'], np.arange(1, len(catalog[r])+1))
                            or not np.array_equal(item['before_y'], target[sl])
                            or not np.array_equal(item['after_y'], final[sl])):
                        raise ValueError(f'{path.name}: plotting motif IDs or values do not match.')
            if group == 'empirical':
                if path.name not in real_map:
                    raise ValueError(f'No unique original network identity for {path.name}.')
                dataset, item, original_hash = real_map[path.name]
                fields = dict(line.split(': ', 1) for line in
                              (data_root / dataset / 'description.txt').read_text(encoding='utf-8-sig').splitlines()
                              if ': ' in line)
                identity = int(fields['Dataset ID'])
                cardinalities = ast.literal_eval(fields['Hyperedge cardinality counts'])
                if (original_hash != sha or (item['N'], item['M']) != (n, m)
                        or int(fields['Node count']) != n
                        or sum(v for k, v in cardinalities.items() if 2 <= int(k) <= 6) != m):
                    raise ValueError(f'Empirical identity/size/source mismatch: {path.name}')
                name, category = fields['Source dataset name'], fields['Network type code']
                if name in ('ECO-M_PL_015_plants', 'ECO-M_PL_062_plants'):
                    category = 'ecology'  # User-confirmed plant-pollinator semantics.
            else:
                identity, name, category, dataset = generation, path.stem, 'artificial', ''
            if identity in seen_ids:
                raise ValueError(f'Duplicate {group} network ID: {identity}')
            seen_ids.add(identity)
            metrics, active = _func_hm_reconstruction_metrics(target, final, keys, active_threshold)
            row = dict(id=identity, name=name, category=category, dataset=dataset, N=n, M=m,
                       generation=generation, metrics=metrics, active_counts=active,
                       source_file=f'{folder}/{path.name}', sha256=sha)
            groups[group].append(row)
            source_checks.append((str(path), sha))
            for r, sl in [(2, slice(0, 50)), (3, slice(50, 1496))]:
                tar, rec = target[sl], final[sl]
                sorted_tar, sorted_rec = np.sort(tar), np.sort(rec)
                for i, key in enumerate(catalog[r]):
                    raw_rows.append([group, identity, r, i+1, str(key), float(tar[i]), float(rec[i]),
                                     float(counts[sl][i]), float(null[sl][i]), float(abs(tar[i]-rec[i])),
                                     bool(abs(tar[i]) >= active_threshold), i+1,
                                     float(sorted_tar[i]), float(sorted_rec[i]),
                                     float(abs(sorted_tar[i]-sorted_rec[i]))])
        groups[group].sort(key=lambda row: row['id'])
    raw_rows.sort(key=lambda row: (0 if row[0] == 'empirical' else 1, row[1], row[2], row[3]))
    summaries = {group: _func_hm_reconstruction_summary(rows) for group, rows in groups.items()}
    payload = dict(groups=groups, summaries=summaries, raw=raw_rows, threshold=float(active_threshold))
    bundle = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = Path(os.environ.get('HM_ARTIFACT_NODE', str(bundle / 'bin/node.exe')))
    modules = Path(os.environ.get('HM_ARTIFACT_MODULES', str(bundle / 'node_modules')))
    if not node.is_file() or not (modules / '@oai/artifact-tool/package.json').is_file():
        raise RuntimeError('Excel export needs Node and @oai/artifact-tool; set HM_ARTIFACT_NODE and HM_ARTIFACT_MODULES.')
    destination = root / '18.reconstruction_sta_cal'
    destination.mkdir(parents=True, exist_ok=True)
    excel_path = destination / 'reconstruction_accuracy.xlsx'
    with tempfile.TemporaryDirectory(prefix='reconstruction-', dir=destination) as temporary:
        temporary = Path(temporary)
        payload_path, temporary_xlsx = temporary / 'data.json', temporary / 'report.xlsx'
        payload_path.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding='utf-8')
        result = subprocess.run([str(node), '--max-old-space-size=8192', '--input-type=module',
                                 '--eval', _HM_RECONSTRUCTION_WORKBOOK_JS, str(payload_path), str(temporary_xlsx)],
                                cwd=str(modules.parent), capture_output=True, text=True, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Workbook export failed: ' + result.stderr[-4000:])
        for source, sha in source_checks:
            if _func_hm_comparison_file_hash(source) != sha:
                raise ValueError(f'Source changed during export: {source}')
        os.replace(temporary_xlsx, excel_path)
    print(f'Reconstruction accuracy: {len(groups["empirical"])} empirical, '
          f'{len(groups["artificial"])} artificial; {len(raw_rows)} motif rows.\n{excel_path}')
    return dict(excel_path=str(excel_path), groups=groups, summaries=summaries, raw_rows=len(raw_rows))
