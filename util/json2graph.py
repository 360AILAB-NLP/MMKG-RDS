import os
import sys
import networkx as nx
import json
import base64
from typing import *
from tqdm import tqdm
from itertools import compress


def build_kg(entities: List[Dict[str, Any]],
             relations: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    根据实体列表和关系列表构建 NetworkX 有向知识图谱
    实体字段: id, name, type, desc, attr
    关系字段: head, relation, tail, attr
    """
    G = nx.DiGraph()
    valid_set = {e['id'] for e in entities}   
    # 加载实体
    for ent in tqdm(entities, desc="node adding to graph"):
        
        # if isinstance(ent, str):
        #     print(ent, isinstance(ent, dict), ent)
        #     continue
        try:
            G.add_node(ent['id'],
                    id=ent['id'],
                    name=ent.get('name', ""),
                    content=ent.get('content', ""),
                    caption=ent.get('caption', ""),
                    img_path=ent.get('img_path',''),
                    type=ent['type'],
                    desc=ent.get('desc', ""),
                    **ent.get('attr', {}))
        except Exception as e:
            print(f"{e}")
            print(f"Error processing entity: {ent}")
    print("node添加完成")
    # 加载关系
    heads_ok = [r["head"] in valid_set for r in relations]  
    tails_ok = [r["tail"] in valid_set for r in relations] 
    mask = [h and t for h, t in zip(heads_ok, tails_ok)]
    good_rels = list(compress(relations, mask))

    entity_ids = [ent['id'] for ent in entities]
    for rel in tqdm(good_rels, desc="relations adding to graph"):
        # if rel['head'] in entity_ids and rel['tail'] in entity_ids:
        G.add_edge(rel['head'],
                rel['tail'],
                relation=rel['relation'],
                desc=rel.get('desc', ""),
                **rel.get('attr', {}))
    print("edge添加完成")
    return G

def save(G, file_name='./output_dir/graph.graphml'):
        G = encode_graph(G)
        nx.write_graphml(G, file_name)
        print(f"NetworkX 图已保存为 {file_name}  节点数={G.number_of_nodes()}  边数={G.number_of_edges()}")

def serialize_lists(G):
    """使用JSON序列化列表属性"""
    # 处理节点属性
    for node, data in G.nodes(data=True):
        for key, value in data.items():
            # if isinstance(value, list):
            data[key] = json.dumps(value, ensure_ascii=False)
    
    # 处理边属性
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            # if isinstance(value, list):
            data[key] = json.dumps(value, ensure_ascii=False)
    
    return G


def encode_graph(G):
    """
    Encode *every* attribute value into base64(json(obj)) so that no special char 
    can break GraphML.
    """
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



from pyvis.network import Network
import json
from typing import List, Dict, Any

def insert_floating_legend(html_content, legend_html):
    """将悬浮图例插入到HTML中"""
    # 在body标签开始后插入图例
    if '<body>' in html_content:
        body_end = html_content.find('<body>') + 6
        html_content = html_content[:body_end] + '\n' + legend_html + '\n' + html_content[body_end:]
    
    return html_content

def visualize_kg_with_legend(entities: str | List[Dict[str, Any]], relations: str | List[Dict[str, Any]], file_name: str = "knowledge_graph.html", vis_node_types: List[str]=["Entity"]):
    """
    简化版的悬浮图例（功能齐全，代码更简洁）
    """
    if isinstance(entities, str):
        with open(entities, 'r', encoding='utf-8') as f:
            entities = json.load(f)
    if isinstance(relations, str):
        with open(relations, 'r', encoding='utf-8') as f:
            relations = json.load(f)
    
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

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
    filtered_nodes = [node for node in entities if node['type'] in vis_node_types]
    node_ids = [ent['id'] for ent in filtered_nodes]
    for ent in filtered_nodes:
        node_id = ent['id']
        node_name = ent['name'][:20]  # 取实体名称的前20个字符作为标签
        node_type = ent['type']
        color = type_colors.get(node_type, "#D3D3D3")  # 默认颜色为灰色
        # 添加节点
        net.add_node(node_id, label=node_name, color=color, title=ent.get('desc', ''))
    # 添加关系边到网络图中
    node_set = set(node_ids)
    rel_set = set([rel['head'] for rel in relations] + [rel['tail'] for rel in relations])
    ex_set = rel_set - node_set
    good_rels = list(filter(lambda x: x['head'] not in ex_set and x['tail'] not in ex_set, relations))


    rel_count = 0
    for rel in good_rels:
        # if rel['head'] in ex_set or rel['tail'] in ex_set: continue
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
    # 生成HTML
    net.write_html(file_name)
    
    # 添加简化悬浮图例
    with open(file_name, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    simple_legend = create_simple_floating_legend(common_colors, len(filtered_nodes), rel_count)
    html_content = insert_floating_legend(html_content, simple_legend)
    
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 知识图谱可视化已生成！节点数目: {len(filtered_nodes)}, 关系数目: {rel_count}, 图例类型: {len(vis_node_types)}, 请打开{file_name}查看。")

    return net

def create_simple_floating_legend(common_colors, entity_count, relation_count):
    """创建简化悬浮图例"""
    legend_items = []
    for entity_type, color in common_colors.items():
        legend_items.append(f'''
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 12px; height: 12px; background: {color}; border-radius: 2px; margin-right: 8px;"></div>
            <span style="font-size: 12px; color: #333;">{entity_type}</span>
        </div>
        ''')
    
    return f'''
    <div id="simple-legend" style="
        position: fixed;
        top: 15px;
        right: 15px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        font-family: Arial;
        max-width: 150px;
    ">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">图例</div>
        {"".join(legend_items)}
        <div style="border-top: 1px solid #eee; margin-top: 8px; padding-top: 8px; font-size: 11px; color: #666;">
            节点: {entity_count}<br>关系: {relation_count}
        </div>
    </div>
    
    <script>
    // 简单拖拽功能
    const legend = document.getElementById('simple-legend');
    let isDragging = false;
    let offset = {{x: 0, y: 0}};
    
    legend.addEventListener('mousedown', (e) => {{
        isDragging = true;
        offset.x = e.clientX - legend.getBoundingClientRect().left;
        offset.y = e.clientY - legend.getBoundingClientRect().top;
        legend.style.cursor = 'grabbing';
    }});
    
    document.addEventListener('mousemove', (e) => {{
        if (!isDragging) return;
        legend.style.left = (e.clientX - offset.x) + 'px';
        legend.style.top = (e.clientY - offset.y) + 'px';
        legend.style.right = 'auto';
    }});
    
    document.addEventListener('mouseup', () => {{
        isDragging = false;
        legend.style.cursor = 'grab';
    }});
    </script>
    '''