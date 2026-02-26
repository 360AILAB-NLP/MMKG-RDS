import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import base64
import networkx as nx
from pyvis.network import Network
from itertools import compress
from typing import *
from tqdm import tqdm


def encode_graph(G):
    def _encode(v):
        blob = json.dumps(v).encode('utf-8') 
        return base64.b64encode(blob).decode('ascii')
    # nodes 
    for _, d in G.nodes(data=True): 
        for k in list(d.keys()): 
            d[k] = _encode(d[k])
    # edges 
    for _, _, d in G.edges(data=True): 
        for k in list(d.keys()): 
            d[k] = _encode(d[k])
    return G 

def decode_graph(G):
    """
    Decode all base64(json(obj)) back to the original Python objects.
    """
    def _decode(b64_str):
        blob = base64.b64decode(b64_str.encode('ascii')) 
        return json.loads(blob.decode('utf-8')) 
    # nodes 
    for _, d in G.nodes(data=True): 
        for k in list(d.keys()): 
            d[k] = _decode(d[k])
    # edges 
    for _, _, d in G.edges(data=True): 
        for k in list(d.keys()): 
            d[k] = _decode(d[k])
    return G 

def load_nx_graphml(file_path: str = 'medical.graphml') -> nx.DiGraph:
    """读取 GraphML 文件，返回 NetworkX 有向图"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"图文件不存在：{file_path}")
    G = nx.read_graphml(file_path)
    G = decode_graph(G)  # 解码
    print(f"图加载完成！节点数={G.number_of_nodes()}，边数={G.number_of_edges()}")
    return G

def build_kg(entities: List[Dict[str, Any]],
             relations: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    根据实体列表和关系列表构建 NetworkX 有向知识图谱
    Args:
        entities: 实体列表, 包含id, name, type, desc, attr等字段
        relations: 关系列表, 包含head, relation, tail, attr等字段
    Reurns:
        G: NetworkX 有向图
    """
    G = nx.DiGraph()
    # 加载实体
    valid_set = {e['id'] for e in entities}   
    for ent in tqdm(entities, desc="加载实体中"):
        G.add_node(ent['id'],
                   id=ent['id'],  # 保留节点ID
                   name=ent['name'],
                   type=ent.get('type', ''),
                   desc=ent.get('desc', ''),
                   content=ent.get('content', ''),
                   caption=ent.get('caption', ''),
                   **ent.get('attr', {}))
    # 加载关系
    heads = [r["head"] in valid_set for r in relations]  
    # print(len(heads_ok))
    tails = [r["tail"] in valid_set for r in relations] 
    # print(len(tails_ok))
    mask = [h and t for h, t in zip(heads, tails)]
    good_rels = list(compress(relations, mask))
    print("过滤后的关系数: ", len(good_rels))
    for rel in tqdm(good_rels, desc='加载关系中'):
        h_id = rel['head']
        t_id = rel['tail']
        G.add_edge(h_id, 
	               t_id,
	               head=h_id,
	               relation=rel['relation'],
	               tail=t_id,
	               **rel.get('attr',  {}))
    return G

def visualize_kg(entities: str | List[Dict[str, Any]], relations: str | List[Dict[str, Any]], file_name: str = "knowledge_graph.html"):
    """
    根据实体和关系列表生成可视化的知识图谱
    """
    if isinstance(entities, str):
        with open(entities, 'r', encoding='utf-8') as f:
            entities = json.load(f)
    if isinstance(relations, str):
        with open(relations, 'r', encoding='utf-8') as f:
            relations = json.load(f)
    # 创建 pyvis 网络图
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    # 获取所有实体类型并为每个类型分配不同的颜色
    entity_types = list(set([node['type'] for node in entities]))
    type_colors = {et: f"#{hex(hash(et) & 0xFFFFFF)[2:].zfill(6)}" for et in entity_types}
    # 更新指定类型的颜色
    common_colors = {
        'Document': '#FF0000',    # 红色
        'Chunk': '#FFA500',       # 橙色
        'Assertion': '#FFFF00',   # 黄色
        'Entity': '#00FF00',      # 绿色
        'Table': '#00FFFF',       # 青色
        'Image': '#0000FF',       # 蓝色 
        'Formula': '#800080'      # 紫色
    }
    type_colors.update(common_colors) 
    # 添加实体节点到网络图中
    node_ids = [ent['id'] for ent in entities]
    for ent in entities:
        entity_id = ent['id']
        entity_name = ent['name'][:20]  # 取实体名称的前20个字符作为标签
        entity_type = ent['type']
        color = type_colors.get(entity_type, "#D3D3D3")  # 默认颜色为灰色
        # 添加节点
        net.add_node(entity_id, label=entity_name, color=color, title=ent.get('desc', ''))
    # 添加关系边到网络图中
    rel_count = 0
    for rel in relations:
        if rel['head'] not in node_ids or rel['tail'] not in node_ids: continue
        rel_count += 1
        # 添加边，使用关系类型作为标签
        net.add_edge(rel['head'], rel['tail'], label=rel['relation'], font={'size': 12, 'color': '#000000'})

    # 设置网络图的一些选项
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "font": {
                "size": 12,
                "face": "Arial"
            }
        }
    }
    """)
    # 生成并保存 HTML 可视化文件
    net.write_html(file_name)
    print(f"✅ 知识图谱已生成！实体数目: {len(entities)}, 关系数目: {rel_count},请打开{file_name}查看。")

def save_kg(G: nx.DiGraph, file_name: str):
    G = encode_graph(G)
    nx.write_graphml(G, file_name)
    print(f"NetworkX 图已保存为 {file_name}  节点数={G.number_of_nodes()}  边数={G.number_of_edges()}")

def nx_to_dict(G):
    # 节点列表（带属性）
    nodes = list(G.nodes(data=True))        # -> [(id, attr_dict), ...]
    # 边列表（带属性）  
    edges = list(G.edges(data=True))        # -> [(u, v, attr_dict), ...]
    return {
            'entities': nodes,
            'relationships': edges,
            'node_count': len(nodes),
            'relationship_count': len(edges),
        }

# 取节点属性
def node_attr(G, name):
    return G.nodes.get(name, {})

# 出边
def out_triples(G, name):
    
    return [(u, d['relation'], v) for u, v, d in G.out_edges(name, data=True)]

# 入边
def in_triples(G, name):
    return [(u, d['relation'], v) for v, u, d in G.in_edges(name, data=True)]

# 获取全部节点类型
def get_node_types(G, key='type'):
    return list(set([node_attr(G, id_)[key] for id_ in G.nodes()]))

# 图中所有三元组
def get_all_relations(G, format="json", key='relation'):
    if format=="tuple":
        return [(u, d.get(key, ""), v) for u, v, d in G.edges(data=True)]
    elif format=="json":
        return [{'head': u, 'relation': d.get('relation', ""), 'tail': v, 'desc': d.get('desc', '')}
        for u, v, d in G.edges(data=True)]
    else:
        raise NotImplementedError

def get_relations(G, name):
    return out_triples(G, name) + in_triples(G, name)

def get_neighbors(G, name):
    # neighbors = [v for (u, r, v) in out_triples(G, name)]
    # neighbors.extend([u for (u, r, v) in out_triples(G, name)])
    return set(G.successors(name)).union(G.predecessors(name))

# 得到宾语
def get_object(G, name, relation) -> str:
    return [v for u, v, d in G.out_edges(name, data=True) if d['relation']==relation][0]



def main():
    nodes = json.load(open("/home/xxxx/xxxx/df_v2/output_dir/exp5_财报/node_list_flat.json", "r"))
    rels = json.load(open("/home/xxxx/xxxx/df_v2/output_dir/exp5_财报/edge_list.json", "r"))
    # G = build_kg(nodes, rels)
    # print("G创建成功!!!")
    # graph_path = "/home/xxxx/xxxx/df_v2/output_dir/exp5_财报/graph.graphml"
    # save_kg(G, file_name=graph_path)  # 在save时已经编码了
    # exit()
    # graph_path = "/home/xxxx/xxxx/df_v2/output_dir/exp4_musique/graph.graphml"
    # G = load_nx_graphml(file_path=graph_path)
    # ent_id = "ent-e143edc709d42c042ed34b84c005738b"
    # node_attr1 = node_attr(G, ent_id)
    # print(node_attr1)
    # relations = get_relations(G, ent_id)
    # for rel in relations:
    #     print(node_attr(G, rel[0])['name'], rel[1], node_attr(G, rel[2])['name'])
    # exit()

    from util.json2graph import visualize_kg_with_legend  # , serialize_lists, encode_graph, decode_graph
    # graph_path = "/home/xxxx/xxxx/df_v2/output_dir/exp3_3pages/graph.graphml"
    # G = load_nx_graphml(file_path=graph_path)

    # ent_id ="ent-770afbc3e73878ead4bcb90bc3720b59"  # DeepDive
    # node_attr1 = node_attr(G, ent_id)
    # print(json.dumps(node_attr1, indent=4, ensure_ascii=False))
    # exit()
    # entities = json.load(open("/home/xxxx/xxxx/df_v2/output_dir/exp3_3pages/node_list_flat_p3.json", "r"))
    # relations = json.load(open("/home/xxxx/xxxx/df_v2/output_dir/exp3_3pages/edge_list_p3.json", "r"))
    # G = build_kg(entities, relations)
    # save_kg(G, file_name=graph_path)
    # exit()
    
    path1 = "/home/xxxx/xxxx/df_v2/output_dir/exp5_财报/graph_legend_tif.html"
    node_types = ['Entity','Table','Image','Formula']  # ['Document','Chunk','Assertion','Entity','Table','Image','Formula']
    visualize_kg_with_legend(nodes, rels, path1, node_types=node_types)
    exit()
    # # G = build_kg(entities, relations)
    ent_id ="ent-770afbc3e73878ead4bcb90bc3720b59"  # DeepDive
    node_attr1 = node_attr(G, ent_id)
    print(node_attr1)
    neighbors = get_neighbors(G, ent_id)
    for nei in neighbors:
        print(node_attr(G, nei).get("name", "None")[: 20])
        print(node_attr(G, nei))
        break
    # attr1 = node_attr1["entity_type"]
    # print(type(attr1))
    # print(attr1)
    # print(type(ast.literal_eval(attr1)))
    # print(json.loads(attr1)[: 2])
    # # 可视化
    # visualize_kg(entities, relations, file_name="/home/xxxx/xxxx/df_v2/output_dir/exp3/graph_deepdive_pages_3.html")
    # visualize_kg([e[1] for e in list(G.nodes(data=True))], [r[2] for r in list(G.edges(data=True))])
    # node_id = "f6643e5f-e517-4707-ac62-2a9821adc19a"
    # print(node_attr(G, node_id)["name"])
    # print(len(get_neighbors(G, node_id)))
    # start_node = max(G.nodes(), key=lambda x: G.out_degree(x))
    # start_name = node_attr(G, start_node)["name"]
    # print(f"首节点: {start_name}")
    # print(G.out_degree( start_node))
    # second_node = max(G.successors(start_node), key=lambda x: G.out_degree(x))
    # second_name = node_attr(G, second_node)["name"]
    # print(second_name)
    # print(G.out_degree(second_node))
    # third_node = max(G.successors(second_node), key=lambda x: G.out_degree(x))
    # third_name = node_attr(G, third_node)["name"]
    # print(third_name)
    # print(G.out_degree(third_node))
    # # 节点属性
    # print(json.dumps(node_attr(G, name=node_id), indent=4, ensure_ascii=False))
    # print("="*20)
    # print(len(get_all_relations(G)))
    # print(get_all_relations(G)[0])
    # # 节点出去的边
    # print(out_triples(G, name=node_id))
    # print(in_triples(G, name=node_id))
    # # 出度
    # print(G.out_degree(node_id)) #  + G.in_degree(node_id))
    # # 入度
    # print(G.in_degree("肺泡蛋白质沉积症"))
    # print("="*20)
    # ob = get_object(G, name="肺泡蛋白质沉积症", relation="belongs_to")
    # print(node_attr(G, name=ob))
    # print(out_triples(G, name=ob))
    # dict1 = nx_to_dict(G=G)
    # dict1 = GraphDict(**dict1)
    # print(dict1.entities[0])
    # print("="*20)
    # print(dict1.node_count)



if __name__ == "__main__":
    main()