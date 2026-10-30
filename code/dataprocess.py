"""数据读取、清洗与格式转换。"""

from py import *


# ------------------------现实世界超图数据------------------------


_REAL_DATA_SCHEMA_VERSION = '2026.09.04-grade-a-real-v1'
_REAL_GRADE_A_FIRST_ID = 241
_REAL_HGX_BASE = 'https://cricca.disi.unitn.it/datasets/hypergraphx-data'
_REAL_HGX_PAGE = 'https://hgx-team.github.io/hypergraphx-data/datasets'
_REAL_TLS_READY = False


def _func_real_spec(name, domain, node_num, edge_num, unique_edge_num,
                    node_semantics, edge_semantics, source = 'hgx',
                    zenodo_record = None, expand_weight = False):
    if source == 'hgx':
        page_url = f'{_REAL_HGX_PAGE}/{name}.html'
        download_url = f'{_REAL_HGX_BASE}/{name}/{name}.json.gz'
        raw_file_name = f'{name}.json.gz'
    else:
        page_url = f'https://zenodo.org/records/{zenodo_record}'
        download_url = f'{page_url}/files/{name}.json'
        raw_file_name = f'{name}.json'
    return {
        'name': name,
        'domain': domain,
        'expected_node_num': int(node_num),
        'expected_edge_num': int(edge_num),
        'expected_unique_edge_num': int(unique_edge_num),
        'node_semantics': node_semantics,
        'edge_semantics': edge_semantics,
        'source_format': source,
        'source_page_url': page_url,
        'download_urls': [download_url],
        'raw_file_name': raw_file_name,
        'expand_weight': bool(expand_weight),
    }


_REAL_GRADE_A_DATASETS = [
    _func_real_spec('mathoverflow-answers', 'community', 73851, 5445, 5445,
                    'MathOverflow questions', 'questions answered by the same user'),
    _func_real_spec('language-country', 'language', 234, 109, 109,
                    'languages', 'languages used in the same country'),
    _func_real_spec('house-committees', 'politics', 1290, 341, 336,
                    'US House legislators', 'members of the same committee',
                    expand_weight = True),
    _func_real_spec('senate-committees', 'politics', 282, 315, 302,
                    'US Senate legislators', 'members of the same committee',
                    expand_weight = True),
    _func_real_spec('walmart-trips', 'consumer', 88860, 65979, 65979,
                    'purchased products', 'products purchased in the same trip'),
    _func_real_spec('plant-pollinator-mpl-034', 'ecology', 26, 128, 63,
                    'plant species', 'pollinator species visiting the same plant',
                    'xgi', 21938218),
    _func_real_spec('plant-pollinator-mpl-049', 'ecology', 37, 225, 118,
                    'plant species', 'pollinator species visiting the same plant',
                    'xgi', 21938279),
    _func_real_spec('ECO-M_PL_015_plants', 'biomedical', 131, 477, 477,
                    'plant species', 'plants visited by the same pollinator'),
    _func_real_spec('ECO-M_PL_015_pollinators', 'biomedical', 666, 129, 129,
                    'pollinator species', 'pollinators visiting the same plant'),
    _func_real_spec('ECO-M_PL_062_pollinators', 'biomedical', 1044, 456, 456,
                    'pollinator species', 'pollinators visiting the same plant'),
    _func_real_spec('movies-cast-action', 'culture', 56699, 6528, 6528,
                    'actors', 'cast members of the same action movie'),
    _func_real_spec('movies-cast-adventure', 'culture', 36503, 3443, 3443,
                    'actors', 'cast members of the same adventure movie'),
    _func_real_spec('movies-cast-comedy', 'culture', 88179, 12581, 12581,
                    'actors', 'cast members of the same comedy movie'),
    _func_real_spec('movies-cast-crime', 'culture', 41001, 4258, 4258,
                    'actors', 'cast members of the same crime movie'),
    _func_real_spec('movies-cast-family', 'culture', 23407, 2658, 2658,
                    'actors', 'cast members of the same family movie'),
    _func_real_spec('movies-cast-fantasy', 'culture', 25146, 2213, 2213,
                    'actors', 'cast members of the same fantasy movie'),
    _func_real_spec('movies-cast-horror', 'culture', 40975, 4583, 4583,
                    'actors', 'cast members of the same horror movie'),
    _func_real_spec('movies-cast-romance', 'culture', 53400, 6679, 6679,
                    'actors', 'cast members of the same romance movie'),
    _func_real_spec('movies-cast-science-fiction', 'culture', 32267, 2971, 2971,
                    'actors', 'cast members of the same science-fiction movie'),
    _func_real_spec('movies-cast-thriller', 'culture', 62024, 7550, 7550,
                    'actors', 'cast members of the same thriller movie'),
    _func_real_spec('coauth-PACS-3', 'collaboration', 43501, 20392, 20392,
                    'scientific authors', 'authors of the same PACS-3 publication'),
    _func_real_spec('coauth-PACS-4', 'collaboration', 71422, 49894, 49894,
                    'scientific authors', 'authors of the same PACS-4 publication'),
    _func_real_spec('coauth-PACS-6', 'collaboration', 93319, 52643, 52643,
                    'scientific authors', 'authors of the same PACS-6 publication'),
    _func_real_spec('coauth-PACS-8', 'collaboration', 78406, 33455, 33455,
                    'scientific authors', 'authors of the same PACS-8 publication'),
    _func_real_spec('cat-edge-Cooking', 'food', 6714, 39243, 39243,
                    'ingredients', 'ingredients used in the same recipe'),
    _func_real_spec('cooking-british', 'food', 1166, 796, 796,
                    'ingredients', 'ingredients used in the same British recipe'),
    _func_real_spec('cooking-cajun_creole', 'food', 1576, 1525, 1525,
                    'ingredients', 'ingredients used in the same Cajun-Creole recipe'),
    _func_real_spec('cooking-chinese', 'food', 1792, 2644, 2644,
                    'ingredients', 'ingredients used in the same Chinese recipe'),
    _func_real_spec('cooking-french', 'food', 2102, 2609, 2609,
                    'ingredients', 'ingredients used in the same French recipe'),
    _func_real_spec('cooking-greek', 'food', 1198, 1162, 1162,
                    'ingredients', 'ingredients used in the same Greek recipe'),
    _func_real_spec('cooking-indian', 'food', 1664, 2980, 2980,
                    'ingredients', 'ingredients used in the same Indian recipe'),
    _func_real_spec('cooking-italian', 'food', 2929, 7758, 7758,
                    'ingredients', 'ingredients used in the same Italian recipe'),
    _func_real_spec('cooking-japanese', 'food', 1439, 1387, 1387,
                    'ingredients', 'ingredients used in the same Japanese recipe'),
    _func_real_spec('cooking-mexican', 'food', 2684, 6376, 6376,
                    'ingredients', 'ingredients used in the same Mexican recipe'),
    _func_real_spec('cooking-southern_us', 'food', 2462, 4235, 4235,
                    'ingredients', 'ingredients used in the same Southern-US recipe'),
    _func_real_spec('cooking-spanish', 'food', 1263, 977, 977,
                    'ingredients', 'ingredients used in the same Spanish recipe'),
    _func_real_spec('cooking-thai', 'food', 1376, 1518, 1518,
                    'ingredients', 'ingredients used in the same Thai recipe'),
    _func_real_spec('whiskyfun', 'food', 208, 117, 117,
                    'flavour descriptors', 'descriptors associated with the same whisky'),
    _func_real_spec('recipe-rec', 'recommendation', 9271, 77733, 74870,
                    'ingredients', 'ingredients used in the same recipe',
                    'xgi', 14003376),
    _func_real_spec('cat-edge-algebra-questions', 'community', 423, 980, 980,
                    'online-community users', 'users associated with the same algebra question'),
    _func_real_spec('cat-edge-geometry-questions', 'community', 580, 888, 888,
                    'online-community users', 'users associated with the same geometry question'),
    _func_real_spec('house-bills', 'politics', 1494, 60987, 54933,
                    'US House legislators', 'sponsors and cosponsors of the same bill',
                    expand_weight = True),
    _func_real_spec('senate-bills', 'politics', 294, 29157, 21721,
                    'US Senate legislators', 'sponsors and cosponsors of the same bill',
                    expand_weight = True),
    _func_real_spec('cat-edge-madison-restaurant-reviews', 'consumer', 565, 594, 594,
                    'reviewers', 'reviewers associated with the same restaurant'),
    _func_real_spec('cat-edge-music-blues-reviews', 'consumer', 1106, 686, 686,
                    'reviewers', 'reviewers associated with the same blues item'),
    _func_real_spec('cat-edge-vegas-bars-reviews', 'consumer', 1234, 1188, 1188,
                    'reviewers', 'reviewers associated with the same Las Vegas bar'),
    _func_real_spec('plant-pollinator-mpl-021', 'ecology', 91, 677, 207,
                    'plant species', 'pollinator species visiting the same plant',
                    'xgi', 21938213),
    _func_real_spec('plant-pollinator-mpl-044', 'ecology', 110, 609, 251,
                    'plant species', 'pollinator species visiting the same plant',
                    'xgi', 21938231),
    _func_real_spec('plant-pollinator-mpl-057', 'ecology', 114, 883, 319,
                    'plant species', 'pollinator species visiting the same plant',
                    'xgi', 21938299),
    _func_real_spec('cat-edge-DAWN', 'biomedical', 2109, 87104, 87104,
                    'drugs', 'drugs recorded in the same emergency-department visit'),
    _func_real_spec('ECO-M_PL_062_plants', 'biomedical', 456, 1008, 1008,
                    'plant species', 'plants visited by the same pollinator'),
    _func_real_spec('Marvel', 'culture', 6439, 10222, 10222,
                    'comic characters', 'characters appearing in the same comic issue'),
    _func_real_spec('movies-cast-animation', 'culture', 9288, 1578, 1578,
                    'actors', 'cast members of the same animation movie'),
    _func_real_spec('coauth-PACS-5', 'collaboration', 17993, 6164, 6164,
                    'scientific authors', 'authors of the same PACS-5 publication'),
    _func_real_spec('wikipedia-coedit', 'collaboration', 8211, 954, 954,
                    'Wikipedia editors', 'editors contributing to the same page'),
    _func_real_spec('cooking-korean', 'food', 898, 818, 818,
                    'ingredients', 'ingredients used in the same Korean recipe'),
    _func_real_spec('cooking-vietnamese', 'food', 1108, 806, 806,
                    'ingredients', 'ingredients used in the same Vietnamese recipe'),
    _func_real_spec('flavourcompounds', 'food', 1107, 1153, 1153,
                    'flavour compounds', 'compounds associated with the same food'),
    _func_real_spec('recipes', 'food', 381, 48983, 48983,
                    'ingredients', 'ingredients used in the same recipe'),
]

for _real_order, _real_dataset_spec in enumerate(_REAL_GRADE_A_DATASETS, 1):
    _real_dataset_spec['report_order'] = _real_order

if len(_REAL_GRADE_A_DATASETS) != 59:
    raise RuntimeError('The grade-A real-world dataset catalog must contain 59 entries.')

_REAL_GRADE_A_BY_NAME = {item['name']: item for item in _REAL_GRADE_A_DATASETS}


def _func_real_safe_text(value):
    return str(value).replace('\r', ' ').replace('\n', ' ').strip()


def _func_real_label_key(value):
    value_type = type(value).__name__
    if isinstance(value, bool):
        return (2, value_type, repr(value))
    if isinstance(value, int):
        return (0, value, value_type)
    if isinstance(value, float) and value.is_integer():
        return (0, int(value), value_type)
    return (1, value_type, str(value))


def _func_real_node_marker(value):
    return type(value).__name__, repr(value)


def _func_real_hash_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _func_real_atomic_json(value, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok = True)
    temp_path = f'{file_path}.tmp'
    with open(temp_path, 'w', encoding = 'utf-8') as file:
        json.dump(value, file, ensure_ascii = False, indent = 2)
    os.replace(temp_path, file_path)


def _func_real_read_description(dataset_path):
    result = {}
    description_path = os.path.join(dataset_path, 'description.txt')
    if not os.path.isfile(description_path):
        return result
    with open(description_path, 'r', encoding = 'utf-8-sig') as file:
        for raw_line in file:
            if ':' not in raw_line:
                continue
            key, value = raw_line.rstrip('\r\n').split(':', 1)
            result[key.strip()] = value.strip()
    return result


def _func_real_existing_dataset(data_path, source_name):
    if not os.path.isdir(data_path):
        return None
    for folder_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(dataset_path) or folder_name.startswith('.'):
            continue
        description = _func_real_read_description(dataset_path)
        if description.get('Source dataset name') == source_name:
            return dataset_path
    return None


def _func_real_next_dataset_id(data_path):
    maximum = 0
    if os.path.isdir(data_path):
        for file_name in os.listdir(data_path):
            match = re.match(r'^(\d+)\.', file_name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _func_real_dataset_id(dataset_path):
    match = re.match(r'^(\d+)\.', os.path.basename(os.path.normpath(dataset_path)))
    return int(match.group(1)) if match else None


def _func_real_assigned_dataset_id(spec):
    return _REAL_GRADE_A_FIRST_ID + int(spec['report_order']) - 1


def _func_real_check_dataset_id(data_path, spec, dataset_id):
    if not os.path.isdir(data_path):
        return
    for folder_name in os.listdir(data_path):
        match = re.match(r'^(\d+)\.', folder_name)
        if not match or int(match.group(1)) != int(dataset_id):
            continue
        dataset_path = os.path.join(data_path, folder_name)
        source_name = _func_real_read_description(dataset_path).get(
            'Source dataset name'
        )
        if source_name != spec['name']:
            raise FileExistsError(
                f'Dataset ID {dataset_id} is already used by {folder_name!r}; '
                f'cannot assign it to {spec["name"]!r}.'
            )


def _func_real_prepare_tls():
    global _REAL_TLS_READY
    if _REAL_TLS_READY:
        return
    if truststore is not None:
        truststore.inject_into_ssl()
    _REAL_TLS_READY = True


def _func_real_download(spec, raw_folder, force = False, timeout = 120):
    os.makedirs(raw_folder, exist_ok = True)
    output_path = os.path.join(raw_folder, spec['raw_file_name'])
    manifest_path = os.path.join(raw_folder, 'download_manifest.json')
    if os.path.isfile(output_path) and not force:
        sha256 = _func_real_hash_file(output_path)
        old_manifest = {}
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, 'r', encoding = 'utf-8') as file:
                    old_manifest = json.load(file)
            except (OSError, ValueError):
                old_manifest = {}
        if not old_manifest.get('sha256') or old_manifest.get('sha256') == sha256:
            old_manifest.update({
                'schema_version': _REAL_DATA_SCHEMA_VERSION,
                'source_dataset_name': spec['name'],
                'source_page_url': spec['source_page_url'],
                'download_url': old_manifest.get(
                    'download_url', spec['download_urls'][0]
                ),
                'raw_file_name': spec['raw_file_name'],
                'sha256': sha256,
                'byte_size': os.path.getsize(output_path),
                'download_complete': True,
            })
            _func_real_atomic_json(old_manifest, manifest_path)
            return output_path, old_manifest

    _func_real_prepare_tls()
    error_list = []
    for url in spec['download_urls']:
        for attempt in range(1, 5):
            part_path = f'{output_path}.part'
            try:
                headers = {'User-Agent': 'HM-real-hypergraph-downloader/1.0'}
                with requests.get(
                    url, stream = True, timeout = (30, int(timeout)),
                    headers = headers, allow_redirects = True,
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get('content-length', 0) or 0)
                    progress = tqdm(
                        total = total or None, unit = 'B', unit_scale = True,
                        unit_divisor = 1024, leave = False,
                        desc = f'Download {spec["name"]}', dynamic_ncols = True,
                    )
                    digest = hashlib.sha256()
                    try:
                        with open(part_path, 'wb') as file:
                            for block in response.iter_content(
                                chunk_size = 1024 * 1024
                            ):
                                if not block:
                                    continue
                                file.write(block)
                                digest.update(block)
                                progress.update(len(block))
                    finally:
                        progress.close()
                os.replace(part_path, output_path)
                manifest = {
                    'schema_version': _REAL_DATA_SCHEMA_VERSION,
                    'source_dataset_name': spec['name'],
                    'source_page_url': spec['source_page_url'],
                    'download_url': url,
                    'raw_file_name': spec['raw_file_name'],
                    'sha256': digest.hexdigest(),
                    'byte_size': os.path.getsize(output_path),
                    'downloaded_at': datetime.now().isoformat(timespec = 'seconds'),
                    'download_complete': True,
                    'processing_complete': False,
                }
                _func_real_atomic_json(manifest, manifest_path)
                return output_path, manifest
            except Exception as error:
                error_list.append(f'{url} (attempt {attempt}): {error}')
                if os.path.isfile(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))
    raise ConnectionError(
        f'Unable to download {spec["name"]} with verified TLS. '
        + ' | '.join(error_list)
    )


def _func_real_load_json(raw_path):
    with open(raw_path, 'rb') as file:
        magic = file.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(raw_path, 'rt', encoding = 'utf-8-sig') as file:
        return json.load(file)


def _func_real_unique_members(member_list):
    result, seen = [], set()
    for node in member_list:
        marker = _func_real_node_marker(node)
        if marker not in seen:
            seen.add(marker)
            result.append(node)
    return result, len(member_list) - len(result)


def _func_real_parse_hgx(spec, raw_path):
    data = _func_real_load_json(raw_path)
    if not isinstance(data, list):
        raise ValueError(f'{spec["name"]}: HypergraphX JSON root must be a list.')
    node_list, edge_list, repeated_membership_num = [], [], 0
    source_metadata = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        if 'hypergraph_type' in item:
            if item['hypergraph_type'] != 'Hypergraph':
                raise ValueError(
                    f'{spec["name"]}: expected an undirected Hypergraph, '
                    f'got {item["hypergraph_type"]!r}.'
                )
            source_metadata = item.get('hypergraph_metadata') or {}
        elif item.get('type') == 'node':
            node_list.append(item['idx'])
        elif item.get('type') == 'edge':
            members, repeated = _func_real_unique_members(
                item.get('interaction', [])
            )
            repeated_membership_num += int(repeated > 0)
            copies = 1
            if spec['expand_weight']:
                weight = (item.get('metadata') or {}).get('weight', 1)
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not float(weight).is_integer()
                    or float(weight) < 1
                ):
                    raise ValueError(
                        f'{spec["name"]}: duplicate-event weight must be a '
                        f'positive integer; got {weight!r}.'
                    )
                copies = int(weight)
            edge_list.extend([members] * copies)
    return node_list, edge_list, repeated_membership_num, source_metadata


def _func_real_parse_xgi(spec, raw_path):
    data = _func_real_load_json(raw_path)
    if not isinstance(data, dict):
        raise ValueError(f'{spec["name"]}: XGI JSON root must be an object.')
    repeated_membership_num = 0
    source_metadata = data.get('metadata') or data.get('hypergraph-data') or {}
    if 'incidences' in data:
        node_list = [
            item.get('node') for item in data.get('nodes', []) if 'node' in item
        ]
        edge_order = [
            item.get('edge') for item in data.get('edges', []) if 'edge' in item
        ]
        edge_member_dict = {edge_id: [] for edge_id in edge_order}
        for incidence in data['incidences']:
            edge_id = incidence['edge']
            if edge_id not in edge_member_dict:
                edge_order.append(edge_id)
                edge_member_dict[edge_id] = []
            edge_member_dict[edge_id].append(incidence['node'])
            node_list.append(incidence['node'])
        edge_list = [edge_member_dict[edge_id] for edge_id in edge_order]
    elif 'edge-dict' in data:
        node_list = list((data.get('node-data') or {}).keys())
        edge_list = list(data['edge-dict'].values())
    else:
        raise ValueError(
            f'{spec["name"]}: XGI JSON contains neither incidences nor edge-dict.'
        )

    normalized_edge_list = []
    for members in edge_list:
        unique_members, repeated = _func_real_unique_members(list(members))
        repeated_membership_num += int(repeated > 0)
        normalized_edge_list.append(unique_members)
    return node_list, normalized_edge_list, repeated_membership_num, source_metadata


def _func_real_parse(spec, raw_path):
    if spec['source_format'] == 'hgx':
        value = _func_real_parse_hgx(spec, raw_path)
    else:
        value = _func_real_parse_xgi(spec, raw_path)
    node_list, edge_list, repeated_membership_num, source_metadata = value
    unique_node_dict = {}
    for node in chain(node_list, chain.from_iterable(edge_list)):
        unique_node_dict.setdefault(_func_real_node_marker(node), node)
    node_list = sorted(unique_node_dict.values(), key = _func_real_label_key)
    return node_list, edge_list, repeated_membership_num, source_metadata


def _func_real_statistics(node_list, edge_list):
    normalized = [
        tuple(sorted(edge, key = _func_real_label_key)) for edge in edge_list
    ]
    cardinality = Counter(len(edge) for edge in normalized)
    unique_edge_num = len({
        tuple(_func_real_node_marker(node) for node in edge)
        for edge in normalized
    })
    return {
        'node_num': len(node_list),
        'edge_num': len(normalized),
        'unique_edge_num': unique_edge_num,
        'duplicate_edge_num': len(normalized) - unique_edge_num,
        'incidence_num': sum(map(len, normalized)),
        'cardinality_counts': dict(sorted(cardinality.items())),
    }


def _func_real_report_mismatch(spec, raw_statistics):
    comparison = {
        'node_num': (raw_statistics['node_num'], spec['expected_node_num']),
        'edge_num': (raw_statistics['edge_num'], spec['expected_edge_num']),
        'unique_edge_num': (
            raw_statistics['unique_edge_num'], spec['expected_unique_edge_num']
        ),
    }
    mismatch = {
        key: {'downloaded': value[0], 'report': value[1]}
        for key, value in comparison.items() if value[0] != value[1]
    }
    if mismatch:
        warnings.warn(
            f'{spec["name"]}: downloaded version differs from the report: '
            f'{mismatch}', RuntimeWarning,
        )
    return mismatch


def _func_real_write_dataset(dataset_path, dataset_id, dataset_name, spec,
                             raw_path, download_manifest, node_list,
                             raw_edge_list, repeated_membership_num,
                             source_metadata, mismatch):
    node_map = {
        _func_real_node_marker(node): index for index, node in enumerate(node_list)
    }
    clean_edge_list = []
    removed_empty_num = 0
    removed_singleton_num = 0
    for edge in raw_edge_list:
        if len(edge) == 0:
            removed_empty_num += 1
        elif len(edge) == 1:
            removed_singleton_num += 1
        else:
            clean_edge_list.append(tuple(
                node_map[_func_real_node_marker(node)] for node in edge
            ))

    raw_statistics = _func_real_statistics(node_list, raw_edge_list)
    clean_statistics = _func_real_statistics(
        list(range(len(node_list))), clean_edge_list
    )
    os.makedirs(dataset_path, exist_ok = True)

    link_path = os.path.join(dataset_path, 'link.csv')
    link_temp_path = f'{link_path}.tmp'
    with open(link_temp_path, 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['hyperedge_id', 'node_id'])
        for edge_id, hyperedge in enumerate(clean_edge_list):
            for node_id in hyperedge:
                writer.writerow([edge_id, node_id])
    os.replace(link_temp_path, link_path)

    mapping_path = os.path.join(dataset_path, 'node_mapping.csv')
    mapping_temp_path = f'{mapping_path}.tmp'
    with open(mapping_temp_path, 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['node_id', 'original_node_id', 'original_node_type'])
        for node_id, original_node in enumerate(node_list):
            writer.writerow([
                node_id,
                json.dumps(original_node, ensure_ascii = False),
                type(original_node).__name__,
            ])
    os.replace(mapping_temp_path, mapping_path)

    source_version = ''
    if isinstance(source_metadata, dict):
        source_version = source_metadata.get('version', '')
    link_sha256 = _func_real_hash_file(link_path)
    description_path = os.path.join(dataset_path, 'description.txt')
    description_temp_path = f'{description_path}.tmp'
    with open(description_temp_path, 'w', encoding = 'utf-8') as file:
        file.write(f'Dataset ID: {dataset_id}\n')
        file.write(f'Dataset folder: {dataset_name}\n')
        file.write('Dataset class: real\n')
        file.write(f'Network type code: {spec["domain"]}\n')
        file.write(f'Network model: Real-world {spec["domain"]} hypergraph\n')
        file.write(f'Node count: {clean_statistics["node_num"]}\n')
        file.write(f'Hyperedge count: {clean_statistics["edge_num"]}\n')
        file.write(f'Incidence count: {clean_statistics["incidence_num"]}\n')
        file.write(
            f'Hyperedge cardinality counts: '
            f'{clean_statistics["cardinality_counts"]}\n'
        )
        file.write(
            f'Duplicate hyperedge count: '
            f'{clean_statistics["duplicate_edge_num"]}\n'
        )
        file.write('Hyperedges containing repeated node memberships: 0\n')
        file.write(f'Source dataset name: {spec["name"]}\n')
        file.write('Report grade: A\n')
        file.write(f'Report order: {spec["report_order"]}\n')
        file.write(
            f'Node semantics: {_func_real_safe_text(spec["node_semantics"])}\n'
        )
        file.write(
            f'Hyperedge semantics: '
            f'{_func_real_safe_text(spec["edge_semantics"])}\n'
        )
        file.write(f'Source format: {spec["source_format"]}\n')
        file.write(f'Source page URL: {spec["source_page_url"]}\n')
        file.write(f'Download URL: {download_manifest["download_url"]}\n')
        file.write(
            f'Source version: '
            f'{_func_real_safe_text(source_version) or "not stated"}\n'
        )
        file.write(f'Raw file: raw/{os.path.basename(raw_path)}\n')
        file.write(f'Raw file SHA-256: {download_manifest["sha256"]}\n')
        file.write(f'Raw node count: {raw_statistics["node_num"]}\n')
        file.write(f'Raw hyperedge count: {raw_statistics["edge_num"]}\n')
        file.write(
            f'Raw unique hyperedge count: '
            f'{raw_statistics["unique_edge_num"]}\n'
        )
        file.write(f'Raw incidence count: {raw_statistics["incidence_num"]}\n')
        file.write(
            f'Raw cardinality counts: {raw_statistics["cardinality_counts"]}\n'
        )
        file.write(
            f'Raw duplicate hyperedge count: '
            f'{raw_statistics["duplicate_edge_num"]}\n'
        )
        file.write(f'Removed empty hyperedges: {removed_empty_num}\n')
        file.write(f'Removed singleton hyperedges: {removed_singleton_num}\n')
        file.write(
            f'Raw hyperedges with repeated node memberships: '
            f'{repeated_membership_num}\n'
        )
        file.write(f'Report statistics mismatch: {mismatch}\n')
        file.write(f'link.csv SHA-256: {link_sha256}\n')
        file.write(f'Processing schema: {_REAL_DATA_SCHEMA_VERSION}\n')
        file.write(
            f'Processed at: {datetime.now().isoformat(timespec = "seconds")}\n'
        )
        file.write('Processing complete: True\n')
        file.write(
            'Cleaning rule: remove empty/singleton hyperedges after removing '
            'within-edge repeated memberships; preserve large and duplicate '
            'hyperedges.\n'
        )
        file.write(
            'link.csv format: each row is one (hyperedge_id, node_id) incidence.\n'
        )
    os.replace(description_temp_path, description_path)
    return raw_statistics, clean_statistics, link_sha256


def _func_real_validate_output(dataset_path, source_name = None):
    description = _func_real_read_description(dataset_path)
    if description.get('Processing complete') != 'True':
        raise ValueError(f'Incomplete description: {dataset_path}')
    if source_name and description.get('Source dataset name') != source_name:
        raise ValueError(f'Unexpected source dataset in {dataset_path}.')
    node_num = int(description.get('Node count', -1))
    expected_edge_num = int(description.get('Hyperedge count', -1))
    expected_incidence_num = int(description.get('Incidence count', -1))
    link_path = os.path.join(dataset_path, 'link.csv')
    edge_members = defaultdict(list)
    incidence_num = 0
    with open(link_path, 'r', newline = '', encoding = 'utf-8-sig') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ['hyperedge_id', 'node_id']:
            raise ValueError(f'Invalid link.csv header: {link_path}')
        for row in reader:
            edge_id = int(row['hyperedge_id'])
            node_id = int(row['node_id'])
            if not 0 <= node_id < node_num:
                raise ValueError(f'Node ID out of range in {link_path}: {node_id}')
            edge_members[edge_id].append(node_id)
            incidence_num += 1
    edge_ids = sorted(edge_members)
    if edge_ids != list(range(expected_edge_num)):
        raise ValueError(f'Hyperedge IDs are not contiguous in {link_path}.')
    for edge_id, members in edge_members.items():
        if len(members) < 2 or len(members) != len(set(members)):
            raise ValueError(f'Invalid cleaned hyperedge {edge_id} in {link_path}.')
    if incidence_num != expected_incidence_num:
        raise ValueError(f'Incidence count mismatch in {link_path}.')

    mapping_path = os.path.join(dataset_path, 'node_mapping.csv')
    with open(mapping_path, 'r', newline = '', encoding = 'utf-8-sig') as file:
        mapping_ids = [int(row['node_id']) for row in csv.DictReader(file)]
    if mapping_ids != list(range(node_num)):
        raise ValueError(f'Invalid node mapping in {mapping_path}.')
    link_sha256 = _func_real_hash_file(link_path)
    if description.get('link.csv SHA-256') != link_sha256:
        raise ValueError(f'link.csv hash mismatch in {dataset_path}.')
    return {
        'dataset_path': dataset_path,
        'node_num': node_num,
        'edge_num': expected_edge_num,
        'incidence_num': incidence_num,
        'link_sha256': link_sha256,
    }


def _func_real_is_complete(dataset_path, spec):
    try:
        validation = _func_real_validate_output(dataset_path, spec['name'])
        manifest_path = os.path.join(
            dataset_path, 'raw', 'download_manifest.json'
        )
        with open(manifest_path, 'r', encoding = 'utf-8') as file:
            manifest = json.load(file)
        raw_path = os.path.join(dataset_path, 'raw', manifest['raw_file_name'])
        return (
            manifest.get('processing_complete') is True
            and manifest.get('schema_version') == _REAL_DATA_SCHEMA_VERSION
            and manifest.get('source_dataset_name') == spec['name']
            and os.path.isfile(raw_path)
            and _func_real_hash_file(raw_path) == manifest.get('sha256')
            and validation['link_sha256'] == manifest.get('link_sha256')
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _func_real_process_dataset(spec, data_path, dataset_id = None,
                               force = False, timeout = 120):
    os.makedirs(data_path, exist_ok = True)
    existing_path = _func_real_existing_dataset(data_path, spec['name'])
    if existing_path and not force and _func_real_is_complete(existing_path, spec):
        validation = _func_real_validate_output(existing_path, spec['name'])
        validation['status'] = 'skipped'
        return validation

    if existing_path:
        dataset_id = _func_real_dataset_id(existing_path)
        working_path = existing_path
    else:
        dataset_id = int(dataset_id or _func_real_assigned_dataset_id(spec))
        _func_real_check_dataset_id(data_path, spec, dataset_id)
        staging_root = os.path.join(data_path, '.real_data_staging')
        working_path = os.path.join(staging_root, spec['name'])
        os.makedirs(working_path, exist_ok = True)

    raw_folder = os.path.join(working_path, 'raw')
    raw_path, download_manifest = _func_real_download(
        spec, raw_folder, force = force, timeout = timeout
    )
    node_list, raw_edge_list, repeated_membership_num, source_metadata = (
        _func_real_parse(spec, raw_path)
    )
    raw_statistics = _func_real_statistics(node_list, raw_edge_list)
    mismatch = _func_real_report_mismatch(spec, raw_statistics)
    dataset_name = (
        f'{dataset_id}.r_{spec["domain"]}_'
        f'{raw_statistics["node_num"]}_{spec["name"]}'
    )
    raw_statistics, clean_statistics, link_sha256 = _func_real_write_dataset(
        working_path, dataset_id, dataset_name, spec, raw_path,
        download_manifest, node_list, raw_edge_list, repeated_membership_num,
        source_metadata, mismatch,
    )
    download_manifest.update({
        'processing_complete': True,
        'processed_at': datetime.now().isoformat(timespec = 'seconds'),
        'processing_schema': _REAL_DATA_SCHEMA_VERSION,
        'dataset_folder': dataset_name,
        'raw_statistics': raw_statistics,
        'analysis_ready_statistics': clean_statistics,
        'report_statistics_mismatch': mismatch,
        'link_sha256': link_sha256,
    })
    _func_real_atomic_json(
        download_manifest, os.path.join(raw_folder, 'download_manifest.json')
    )
    _func_real_validate_output(working_path, spec['name'])

    final_path = os.path.join(data_path, dataset_name)
    if os.path.normcase(os.path.abspath(working_path)) != os.path.normcase(
        os.path.abspath(final_path)
    ):
        if os.path.exists(final_path):
            raise FileExistsError(f'Target dataset folder already exists: {final_path}')
        os.replace(working_path, final_path)
        staging_root = os.path.dirname(working_path)
        try:
            os.rmdir(staging_root)
        except OSError:
            pass
    validation = _func_real_validate_output(final_path, spec['name'])
    validation.update({
        'status': 'processed',
        'report_statistics_mismatch': mismatch,
        'raw_statistics': raw_statistics,
        'analysis_ready_statistics': clean_statistics,
    })
    return validation


def _func_real_process_named(name, data_path, dataset_id = None,
                             force = False, timeout = 120):
    spec = _REAL_GRADE_A_BY_NAME[name]
    dataset_id = dataset_id or _func_real_assigned_dataset_id(spec)
    return _func_real_process_dataset(spec, data_path, dataset_id, force, timeout)


def process_all_grade_a_real_hypergraphs(data_path: str, force = False,
                                         timeout = 120):
    """下载并转换报告中的全部 59 个 A 级现实世界超图，支持断点续算。"""
    result = {}
    progress = tqdm(
        _REAL_GRADE_A_DATASETS, desc = 'Grade-A real hypergraphs',
        unit = 'network', dynamic_ncols = True,
    )
    for spec in progress:
        progress.set_postfix_str(spec['name'])
        existing_path = _func_real_existing_dataset(data_path, spec['name'])
        assigned_id = (
            _func_real_dataset_id(existing_path)
            if existing_path else _func_real_assigned_dataset_id(spec)
        )
        try:
            result[spec['name']] = _func_real_process_dataset(
                spec, data_path, assigned_id, force, timeout
            )
        except Exception as error:
            result[spec['name']] = {
                'status': 'failed',
                'error': f'{type(error).__name__}: {error}',
            }
            tqdm.write(f'[FAILED] {spec["name"]}: {error}')
    progress.close()
    failed = [
        name for name, value in result.items() if value['status'] == 'failed'
    ]
    summary_path = os.path.join(
        data_path, '_grade_a_real_processing_summary.json'
    )
    _func_real_atomic_json({
        'schema_version': _REAL_DATA_SCHEMA_VERSION,
        'complete': not failed,
        'dataset_num': len(_REAL_GRADE_A_DATASETS),
        'failed': failed,
        'results': result,
        'updated_at': datetime.now().isoformat(timespec = 'seconds'),
    }, summary_path)
    if failed:
        raise RuntimeError(
            f'{len(failed)} grade-A datasets failed; see {summary_path}: {failed}'
        )
    return result


def validate_all_grade_a_real_hypergraphs(data_path: str):
    """严格回读并校验已经完成的 A 级现实世界超图。"""
    result = {}
    for spec in _REAL_GRADE_A_DATASETS:
        dataset_path = _func_real_existing_dataset(data_path, spec['name'])
        if dataset_path is None:
            raise FileNotFoundError(f'Missing processed dataset: {spec["name"]}')
        result[spec['name']] = _func_real_validate_output(
            dataset_path, spec['name']
        )
    return result


def real_world_hypergraph_data_self_check():
    """用小型 HGX、加权 HGX、XGI legacy 和 HIF 夹具检查解析口径。"""
    with tempfile.TemporaryDirectory() as folder:
        plain_spec = dict(_REAL_GRADE_A_DATASETS[0])
        plain_spec.update({'name': 'plain', 'expand_weight': False})
        plain_path = os.path.join(folder, 'plain.json.gz')
        with gzip.open(plain_path, 'wt', encoding = 'utf-8') as file:
            json.dump([
                {'hypergraph_type': 'Hypergraph', 'hypergraph_metadata': {}},
                {'type': 'node', 'idx': 'isolated', 'metadata': {}},
                {'type': 'edge', 'interaction': ['a', 'b'], 'metadata': {}},
                {'type': 'edge', 'interaction': ['c'], 'metadata': {}},
            ], file)
        nodes, edges, repeated, _ = _func_real_parse_hgx(
            plain_spec, plain_path
        )
        assert len(nodes) == 1 and len(edges) == 2 and repeated == 0

        weighted_spec = dict(plain_spec)
        weighted_spec.update({'name': 'weighted', 'expand_weight': True})
        weighted_path = os.path.join(folder, 'weighted.json')
        with open(weighted_path, 'w', encoding = 'utf-8') as file:
            json.dump([
                {
                    'hypergraph_type': 'Hypergraph',
                    'hypergraph_metadata': {'weighted': True},
                },
                {
                    'type': 'edge', 'interaction': [0, 1],
                    'metadata': {'weight': 3},
                },
            ], file)
        _, weighted_edges, _, _ = _func_real_parse_hgx(
            weighted_spec, weighted_path
        )
        assert weighted_edges == [[0, 1], [0, 1], [0, 1]]

        legacy_path = os.path.join(folder, 'legacy.json')
        with open(legacy_path, 'w', encoding = 'utf-8') as file:
            json.dump({
                'hypergraph-data': {},
                'node-data': {'0': {}, '1': {}, '2': {}},
                'edge-dict': {'a': ['0', '1'], 'b': ['0', '1']},
            }, file)
        legacy_spec = dict(plain_spec)
        legacy_spec['source_format'] = 'xgi'
        legacy_nodes, legacy_edges, _, _ = _func_real_parse_xgi(
            legacy_spec, legacy_path
        )
        assert len(legacy_nodes) == 3 and len(legacy_edges) == 2

        hif_path = os.path.join(folder, 'hif.json')
        with open(hif_path, 'w', encoding = 'utf-8') as file:
            json.dump({
                'network-type': 'undirected',
                'nodes': [{'node': 0}, {'node': 1}, {'node': 2}],
                'edges': [{'edge': 'a'}, {'edge': 'b'}],
                'incidences': [
                    {'edge': 'a', 'node': 0},
                    {'edge': 'a', 'node': 1},
                    {'edge': 'a', 'node': 1},
                    {'edge': 'b', 'node': 2},
                ],
            }, file)
        hif_nodes, hif_edges, hif_repeated, _ = _func_real_parse_xgi(
            legacy_spec, hif_path
        )
        assert len(hif_nodes) >= 3 and hif_edges == [[0, 1], [2]]
        assert hif_repeated == 1

        output_spec = dict(plain_spec)
        output_spec.update({
            'name': 'fixture-output',
            'domain': 'ecology',
            'report_order': 1,
        })
        output_raw_folder = os.path.join(folder, 'output', 'raw')
        os.makedirs(output_raw_folder, exist_ok = True)
        output_raw_path = os.path.join(output_raw_folder, 'fixture.json')
        with open(output_raw_path, 'w', encoding = 'utf-8') as file:
            json.dump([
                {'hypergraph_type': 'Hypergraph', 'hypergraph_metadata': {}},
                {'type': 'node', 'idx': 'isolated', 'metadata': {}},
                {
                    'type': 'edge', 'interaction': ['a', 'b', 'b'],
                    'metadata': {},
                },
                {'type': 'edge', 'interaction': ['a', 'b'], 'metadata': {}},
                {'type': 'edge', 'interaction': ['c'], 'metadata': {}},
                {'type': 'edge', 'interaction': [], 'metadata': {}},
                {
                    'type': 'edge', 'interaction': ['a', 'b', 'c', 'd'],
                    'metadata': {},
                },
            ], file)
        output_nodes, output_edges, output_repeated, output_metadata = (
            _func_real_parse(output_spec, output_raw_path)
        )
        output_manifest = {
            'download_url': 'fixture://local',
            'sha256': _func_real_hash_file(output_raw_path),
        }
        output_path = os.path.join(folder, 'output')
        _, clean_statistics, _ = _func_real_write_dataset(
            output_path, 241, '241.r_ecology_5_fixture-output', output_spec,
            output_raw_path, output_manifest, output_nodes, output_edges,
            output_repeated, output_metadata, {},
        )
        validation = _func_real_validate_output(output_path, 'fixture-output')
        output_description = _func_real_read_description(output_path)
        assert validation['node_num'] == 5 and validation['edge_num'] == 3
        assert validation['incidence_num'] == 8
        assert clean_statistics['duplicate_edge_num'] == 1
        assert output_description['Removed empty hyperedges'] == '1'
        assert output_description['Removed singleton hyperedges'] == '1'
        assert output_description['Raw hyperedges with repeated node memberships'] == '1'
    return True


# 每个数据集保留独立公开入口；底层下载、清洗和验证逻辑由共享方法实现。
def process_mathoverflow_answers(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('mathoverflow-answers', data_path, dataset_id, force, timeout)
def process_language_country(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('language-country', data_path, dataset_id, force, timeout)
def process_house_committees(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('house-committees', data_path, dataset_id, force, timeout)
def process_senate_committees(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('senate-committees', data_path, dataset_id, force, timeout)
def process_walmart_trips(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('walmart-trips', data_path, dataset_id, force, timeout)
def process_plant_pollinator_mpl_034(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('plant-pollinator-mpl-034', data_path, dataset_id, force, timeout)
def process_plant_pollinator_mpl_049(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('plant-pollinator-mpl-049', data_path, dataset_id, force, timeout)
def process_eco_m_pl_015_plants(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('ECO-M_PL_015_plants', data_path, dataset_id, force, timeout)
def process_eco_m_pl_015_pollinators(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('ECO-M_PL_015_pollinators', data_path, dataset_id, force, timeout)
def process_eco_m_pl_062_pollinators(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('ECO-M_PL_062_pollinators', data_path, dataset_id, force, timeout)
def process_movies_cast_action(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-action', data_path, dataset_id, force, timeout)
def process_movies_cast_adventure(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-adventure', data_path, dataset_id, force, timeout)
def process_movies_cast_comedy(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-comedy', data_path, dataset_id, force, timeout)
def process_movies_cast_crime(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-crime', data_path, dataset_id, force, timeout)
def process_movies_cast_family(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-family', data_path, dataset_id, force, timeout)
def process_movies_cast_fantasy(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-fantasy', data_path, dataset_id, force, timeout)
def process_movies_cast_horror(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-horror', data_path, dataset_id, force, timeout)
def process_movies_cast_romance(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-romance', data_path, dataset_id, force, timeout)
def process_movies_cast_science_fiction(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-science-fiction', data_path, dataset_id, force, timeout)
def process_movies_cast_thriller(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-thriller', data_path, dataset_id, force, timeout)
def process_coauth_pacs_3(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('coauth-PACS-3', data_path, dataset_id, force, timeout)
def process_coauth_pacs_4(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('coauth-PACS-4', data_path, dataset_id, force, timeout)
def process_coauth_pacs_6(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('coauth-PACS-6', data_path, dataset_id, force, timeout)
def process_coauth_pacs_8(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('coauth-PACS-8', data_path, dataset_id, force, timeout)
def process_cat_edge_cooking(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-Cooking', data_path, dataset_id, force, timeout)
def process_cooking_british(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-british', data_path, dataset_id, force, timeout)
def process_cooking_cajun_creole(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-cajun_creole', data_path, dataset_id, force, timeout)
def process_cooking_chinese(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-chinese', data_path, dataset_id, force, timeout)
def process_cooking_french(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-french', data_path, dataset_id, force, timeout)
def process_cooking_greek(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-greek', data_path, dataset_id, force, timeout)
def process_cooking_indian(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-indian', data_path, dataset_id, force, timeout)
def process_cooking_italian(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-italian', data_path, dataset_id, force, timeout)
def process_cooking_japanese(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-japanese', data_path, dataset_id, force, timeout)
def process_cooking_mexican(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-mexican', data_path, dataset_id, force, timeout)
def process_cooking_southern_us(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-southern_us', data_path, dataset_id, force, timeout)
def process_cooking_spanish(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-spanish', data_path, dataset_id, force, timeout)
def process_cooking_thai(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-thai', data_path, dataset_id, force, timeout)
def process_whiskyfun(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('whiskyfun', data_path, dataset_id, force, timeout)
def process_recipe_rec(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('recipe-rec', data_path, dataset_id, force, timeout)
def process_cat_edge_algebra_questions(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-algebra-questions', data_path, dataset_id, force, timeout)
def process_cat_edge_geometry_questions(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-geometry-questions', data_path, dataset_id, force, timeout)
def process_house_bills(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('house-bills', data_path, dataset_id, force, timeout)
def process_senate_bills(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('senate-bills', data_path, dataset_id, force, timeout)
def process_cat_edge_madison_restaurant_reviews(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-madison-restaurant-reviews', data_path, dataset_id, force, timeout)
def process_cat_edge_music_blues_reviews(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-music-blues-reviews', data_path, dataset_id, force, timeout)
def process_cat_edge_vegas_bars_reviews(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-vegas-bars-reviews', data_path, dataset_id, force, timeout)
def process_plant_pollinator_mpl_021(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('plant-pollinator-mpl-021', data_path, dataset_id, force, timeout)
def process_plant_pollinator_mpl_044(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('plant-pollinator-mpl-044', data_path, dataset_id, force, timeout)
def process_plant_pollinator_mpl_057(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('plant-pollinator-mpl-057', data_path, dataset_id, force, timeout)
def process_cat_edge_dawn(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cat-edge-DAWN', data_path, dataset_id, force, timeout)
def process_eco_m_pl_062_plants(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('ECO-M_PL_062_plants', data_path, dataset_id, force, timeout)
def process_marvel(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('Marvel', data_path, dataset_id, force, timeout)
def process_movies_cast_animation(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('movies-cast-animation', data_path, dataset_id, force, timeout)
def process_coauth_pacs_5(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('coauth-PACS-5', data_path, dataset_id, force, timeout)
def process_wikipedia_coedit(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('wikipedia-coedit', data_path, dataset_id, force, timeout)
def process_cooking_korean(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-korean', data_path, dataset_id, force, timeout)
def process_cooking_vietnamese(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('cooking-vietnamese', data_path, dataset_id, force, timeout)
def process_flavourcompounds(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('flavourcompounds', data_path, dataset_id, force, timeout)
def process_recipes(data_path, dataset_id = None, force = False, timeout = 120): return _func_real_process_named('recipes', data_path, dataset_id, force, timeout)
