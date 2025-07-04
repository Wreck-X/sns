import os, re, random, argparse, pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
from email.parser import Parser
from sklearn.preprocessing import minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

# Set better matplotlib defaults
plt.style.use('default')
sns.set_palette("husl")

# Load ML models
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('rf_phishing_classifier.pkl', 'rb') as f:
        rf_classifier = pickle.load(f)
    ML_MODELS_LOADED = True
    print("[ML] Successfully loaded TF-IDF vectorizer and Random Forest classifier")
except Exception as e:
    print(f"[ML] Failed to load ML models: {e}. Falling back to rule-based detection")
    tfidf_vectorizer = None
    rf_classifier = None
    ML_MODELS_LOADED = False

# ---------------------------------------------------------
# EMAIL PARSING & GRAPH BUILD
# ---------------------------------------------------------
def parse_emails(directory: str):
    out = []
    for root, _, files in os.walk(directory):
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='latin1') as fh:
                    msg = Parser().parsestr(fh.read())
                    body = msg.get_payload()
                    if isinstance(body, list):
                        body = ' '.join(map(str, body))
                    out.append({'from': msg.get('from', '').strip(),
                                'to': msg.get('to', '').strip(),
                                'body': body or ''})
            except:
                continue
    return out

def build_graph(emails: List[dict]):
    G = nx.DiGraph()
    edge_weights = defaultdict(int)
    
    for e in emails:
        s = e['from']
        if not s:
            continue
        for r in [x.strip() for x in e['to'].split(',') if x.strip()]:
            G.add_edge(s, r)
            edge_weights[(s, r)] += 1
    
    for (u, v), weight in edge_weights.items():
        if G.has_edge(u, v):
            G[u][v]['weight'] = weight
    
    return G

# ---------------------------------------------------------
# RULE-BASED DETECTORS (kept as fallback)
# ---------------------------------------------------------
SENSITIVE_KWS = [r'\bsalary\b', r'\bbonus\b', r'\bwage\b', r'\bpayroll\b',
                 r'\bssn\b|\bsocial security\b', r'\baccount number\b',
                 r'\brouting number\b', r'\biban\b', r'\bifsc\b', r'\bswift\b',
                 r'\bwire transfer\b', r'\bconfidential\b', r'\binternal use\b', r'\bnda\b']
CREDIT_CARD_RE = re.compile(r'(?:\b\d{4}[-\s]?){3}\d{4}\b')
SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
IBAN_RE = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b')
PHISH_KWS = [(r'\burgent\b', 2), (r'\bverify\b', 2), (r'\bupdate\b', 1),
             (r'\bpassword\b', 2), (r'\bclick\s+here\b', 2), (r'\bsecurity\b', 1),
             (r'\baccount\b', 1), (r'\blogin\b', 1), (r'\bconfirm\b', 1)]
URL_RE = re.compile(r'https?://[^\s]+', re.I)
IP_RE = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}')
PUNY_RE = re.compile(r'https?://xn--', re.I)
RISKY_TLD_RE = re.compile(r'https?://[^\s]+\.(zip|mov|info|ru|cn)(/|$)', re.I)

def is_sensitive(t: str) -> bool:
    l = t.lower()
    if any(re.search(p, l) for p in SENSITIVE_KWS):
        return True
    if CREDIT_CARD_RE.search(t) or SSN_RE.search(t) or IBAN_RE.search(t):
        return True
    return False

def is_phishing_rule_based(t: str, th=3) -> bool:
    l = t.lower()
    score = sum(w for p, w in PHISH_KWS if re.search(p, l))
    if URL_RE.search(l):
        score += 1
    if IP_RE.search(l) or PUNY_RE.search(l) or RISKY_TLD_RE.search(l):
        score += 2
    return score >= th

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    try:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)
    except:
        return text

def is_phishing(t: str, th=3) -> bool:
    if ML_MODELS_LOADED:
        try:
            cleaned_text = clean_text(t)
            X_tfidf = tfidf_vectorizer.transform([cleaned_text])
            meta_features = np.array([[0, 0, 0, 12]])
            X = np.hstack((X_tfidf.toarray(), meta_features))
            prediction = rf_classifier.predict(X)[0]
            return bool(prediction)
        except Exception as e:
            print(f"[ML] Error in ML prediction: {e}. Falling back to rule-based detection")
            return is_phishing_rule_based(t, th)
    else:
        return is_phishing_rule_based(t, th)

# ---------------------------------------------------------
# SEED SELECTION
# ---------------------------------------------------------
def select_seeds(G: nx.DiGraph, k=5, method='degree_discount', eig=False):
    if G.number_of_nodes() == 0:
        return []
    if method == 'degree_discount':
        d = {v: G.out_degree(v) for v in G}
        dd = d.copy()
        S = set()
        while len(S) < k and dd:
            v = max(dd, key=dd.get)
            S.add(v)
            dd.pop(v)
            for u in G.successors(v):
                if u in dd:
                    dd[u] = d[u] - 2 * len([w for w in G.predecessors(u) if w in S]) - len([w for w in G.successors(u) if w in S])
        return list(S)
    if method == 'degree':
        return [n for n, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:k]]
    if method == 'composite':
        deg = nx.out_degree_centrality(G)
        btw = nx.betweenness_centrality(G, k=min(300, G.number_of_nodes()))
        pr = nx.pagerank(G, alpha=0.85)
        data = {'deg': deg, 'btw': btw, 'pr': pr}
        if eig:
            data['eig'] = nx.eigenvector_centrality(G, max_iter=600)
        df = pd.DataFrame(data).apply(minmax_scale)
        df['score'] = df.mean(axis=1)
        return df['score'].nlargest(k).index.tolist()
    return random.sample(list(G.nodes), k)

# ---------------------------------------------------------
# CASCADE
# ---------------------------------------------------------
@dataclass
class CascadeResult:
    active: Set[str]
    timeline: List[int]

def independent_cascade(G: nx.DiGraph, seeds: List[str], p=0.1):
    active = set(seeds)
    frontier = set(seeds)
    tl = [len(active)]
    while frontier:
        nxt = {v for u in frontier for v in G.successors(u) if v not in active and random.random() < p}
        active.update(nxt)
        frontier = nxt
        tl.append(len(active))
    return CascadeResult(active, tl)

# ---------------------------------------------------------
# ANOMALY & SOURCE TRACE
# ---------------------------------------------------------
def detect_anomalies(G, emails):
    c = defaultdict(int)
    for e in emails:
        if is_phishing(e['body']) or is_sensitive(e['body']):
            c[e['from']] += 1
    if not c:
        return []
    s = pd.Series(c)
    th = max(3, s.mean() + 2 * s.std())
    return list(s[s > th].index)

def trace_sources(G, inf, top=5):
    sc = defaultdict(float)
    for t in inf:
        for s in G.nodes:
            if nx.has_path(G, s, t):
                sc[s] += 1 / (1 + nx.shortest_path_length(G, s, t))
    return [n for n, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top]]

# ---------------------------------------------------------
# PREVENTIVE MEASURES
# ---------------------------------------------------------
def generate_preventive_measures(emails, anomalies, ph, inf_nodes):
    measures = []
    phishing_count = len(ph)
    sensitive_count = len([e for e in emails if is_sensitive(e['body'])])
    anomaly_count = len(anomalies)
    infected_count = len(set(inf_nodes))

    # User Education
    if phishing_count > 0:
        measures.append("### User Education\n- **Conduct Phishing Awareness Training**: Implement regular training sessions to educate employees on recognizing phishing indicators such as urgent language, suspicious URLs, and requests for sensitive information.\n- **Simulated Phishing Exercises**: Run controlled phishing simulations to test employee responses and reinforce training.\n- **Clear Reporting Channels**: Establish and communicate a straightforward process for employees to report suspected phishing emails.")

    # Email Filtering Enhancements
    if phishing_count > 0 or sensitive_count > 0:
        measures.append("### Email Filtering Enhancements\n- **Update Spam Filters**: Enhance email filters to detect keywords (e.g., 'urgent', 'verify', 'password') and risky TLDs (e.g., .zip, .ru) identified in the pipeline.\n- **Block Suspicious Domains/IPs**: Blacklist domains with IP-based URLs or punycode (xn--) and known risky TLDs.\n- **SPF/DKIM/DMARC Implementation**: Enforce Sender Policy Framework (SPF), DomainKeys Identified Mail (DKIM), and Domain-based Message Authentication, Reporting, and Conformance (DMARC) to prevent email spoofing.")

    # Network-Based Restrictions
    if anomaly_count > 0:
        measures.append(f"### Network-Based Restrictions\n- **Restrict Anomalous Senders**: Identified {anomaly_count} anomalous senders. Consider temporarily restricting email capabilities for accounts: {', '.join(anomalies)} until investigated.\n- **Monitor High-Risk Accounts**: Implement real-time monitoring for accounts with high phishing or sensitive email activity.\n- **Rate Limiting**: Apply rate limits on outgoing emails for accounts exhibiting anomalous behavior to prevent rapid spread.")

    # Incident Response for Infected Nodes
    if infected_count > 0:
        measures.append(f"### Incident Response for Infected Nodes\n- **Quarantine Affected Accounts**: Identified {infected_count} potentially infected recipients. Quarantine these accounts to prevent further spread: {', '.join(list(set(inf_nodes))[:5])}{', ...' if infected_count > 5 else ''}.\n- **Password Resets**: Enforce immediate password resets for affected accounts.\n- **Forensic Analysis**: Conduct a detailed analysis of phishing emails to identify attack vectors and update defenses.")

    # General Security Improvements
    measures.append("### General Security Improvements\n- **Multi-Factor Authentication (MFA)**: Enforce MFA across all email accounts to reduce unauthorized access risks.\n- **Regular Security Audits**: Perform periodic audits of email and network configurations to identify vulnerabilities.\n- **Incident Response Plan**: Develop or update an incident response plan to handle phishing incidents swiftly.")

    # Write measures to a markdown file
    with open('preventive_measures.md', 'w') as f:
        f.write("# Preventive Measures Report\n\n")
        f.write(f"**Summary**: Detected {phishing_count} phishing emails, {sensitive_count} sensitive emails, {anomaly_count} anomalous senders, and {infected_count} infected recipients.\n\n")
        f.write("\n\n".join(measures))
    
    print("[preventive] Saved preventive measures report: preventive_measures.md")
    return measures

# ---------------------------------------------------------
# ENHANCED VISUALISATION
# ---------------------------------------------------------
def get_intelligent_subgraph(G, seeds=None, highlight=None, max_nodes=500):
    seeds = seeds or []
    highlight = highlight or []
    forced = set([n for n in seeds + highlight if n in G])
    
    if G.number_of_nodes() <= max_nodes:
        return G.copy()
    
    try:
        gcc = max(nx.weakly_connected_components(G), key=len)
        subG = G.subgraph(gcc).copy()
    except:
        subG = G.copy()
    
    if subG.number_of_nodes() <= max_nodes:
        return subG
    
    priority_nodes = set(forced)
    degree_nodes = [n for n, d in sorted(subG.degree(), key=lambda x: x[1], reverse=True)]
    priority_nodes.update(degree_nodes[:max_nodes//4])
    
    for node in forced:
        if node in subG:
            priority_nodes.update(subG.neighbors(node))
    
    remaining_quota = max_nodes - len(priority_nodes)
    if remaining_quota > 0:
        candidates = set(subG.nodes()) - priority_nodes
        if candidates:
            additional = random.sample(list(candidates), min(remaining_quota, len(candidates)))
            priority_nodes.update(additional)
    
    return subG.subgraph(list(priority_nodes)).copy()

def get_better_layout(G, layout='spring', spring_k=None, iterations=50):
    if G.number_of_nodes() == 0:
        return {}
    
    if layout == 'kk':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        try:
            pos = nx.spectral_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        k = spring_k if spring_k else 2.0 / np.sqrt(G.number_of_nodes())
        pos = nx.spring_layout(G, seed=42, k=k, iterations=iterations, threshold=1e-4)
    
    return pos

def get_node_sizes(G, seeds=None, highlight=None):
    seeds = seeds or []
    highlight = highlight or []
    
    base_size = 30
    seed_size = 120
    highlight_size = 80
    
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_sizes = {}
    for node in G.nodes():
        if node in seeds:
            size = seed_size
        elif node in highlight:
            size = highlight_size
        else:
            degree_factor = 1 + (degrees.get(node, 0) / max_degree) * 0.5
            size = base_size * degree_factor
        node_sizes[node] = size
    
    return node_sizes

def get_edge_properties(G):
    edge_widths = []
    edge_alphas = []
    
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    
    for u, v in G.edges():
        weight = G[u][v].get('weight', 1)
        width = 0.3 + (weight / max_weight) * 1.2
        edge_widths.append(width)
        alpha = 0.2 + (weight / max_weight) * 0.4
        edge_alphas.append(alpha)
    
    return edge_widths, edge_alphas

def draw_graph(G: nx.DiGraph, seeds=None, highlight=None, max_nodes=500, out='email_graph.png', 
               layout='spring', spring_k=None, colour_comm=False, figsize=(16, 12)):
    if G.number_of_nodes() == 0:
        print("[viz] Empty graph, skipping visualization")
        return
    
    seeds = seeds or []
    highlight = highlight or []
    
    H = get_intelligent_subgraph(G, seeds, highlight, max_nodes)
    print(f"[viz] Using subgraph with {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    
    plt.figure(figsize=figsize, dpi=300)
    plt.clf()
    
    pos = get_better_layout(H, layout, spring_k, iterations=100)
    
    node_colors = ['#e8e8e8'] * H.number_of_nodes()
    if colour_comm and community_louvain and H.number_of_nodes() > 1:
        try:
            part = community_louvain.best_partition(H.to_undirected())
            colors = plt.cm.Set3(np.linspace(0, 1, len(set(part.values()))))
            node_colors = [colors[part[n] % len(colors)] for n in H.nodes()]
        except:
            pass
    
    node_sizes_dict = get_node_sizes(H, [n for n in seeds if n in H], [n for n in highlight if n in H])
    node_sizes = [node_sizes_dict[node] for node in H.nodes()]
    
    edge_widths, edge_alphas = get_edge_properties(H)
    
    if H.number_of_edges() > 0:
        nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.3, edge_color='#666666',
                              arrows=True, arrowsize=8, arrowstyle='->', connectionstyle='arc3,rad=0.05')
    
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors,
                          linewidths=0.5, edgecolors='#333333', alpha=0.8)
    
    seed_nodes_in_graph = [n for n in seeds if n in H]
    if seed_nodes_in_graph:
        seed_sizes = [node_sizes_dict[n] for n in seed_nodes_in_graph]
        nx.draw_networkx_nodes(H, pos, nodelist=seed_nodes_in_graph, node_size=seed_sizes,
                              node_color='#ff6b35', linewidths=2, edgecolors='#d63031', alpha=1.0)
    
    highlight_nodes_in_graph = [n for n in highlight if n in H]
    if highlight_nodes_in_graph:
        highlight_sizes = [node_sizes_dict[n] for n in highlight_nodes_in_graph]
        nx.draw_networkx_nodes(H, pos, nodelist=highlight_nodes_in_graph, node_size=highlight_sizes,
                              node_color='#00b894', linewidths=2, edgecolors='#00a085', alpha=1.0)
    
    if H.number_of_nodes() <= 100:
        important_nodes = set(seed_nodes_in_graph + highlight_nodes_in_graph)
        if important_nodes:
            labels = {n: n.split('@')[0] if '@' in n and len(n) > 15 else n for n in important_nodes}
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight='bold', font_color='#2d3436')
    
    legend_elements = []
    if seed_nodes_in_graph:
        legend_elements.append(patches.Patch(color='#ff6b35', label=f'Seed Nodes ({len(seed_nodes_in_graph)})'))
    if highlight_nodes_in_graph:
        legend_elements.append(patches.Patch(color='#00b894', label=f'Infected Nodes ({len(highlight_nodes_in_graph)})'))
    if colour_comm and community_louvain:
        legend_elements.append(patches.Patch(color='#e8e8e8', label='Communities (colored)'))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.axis('off')
    plt.title(f'Email Network Graph\n{H.number_of_nodes()} nodes, {H.number_of_edges()} edges',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f'[viz] Saved enhanced visualization: {out}')
    
    if H.number_of_nodes() > 200:
        create_simplified_graph(H, seeds, highlight, out.replace('.png', '_simple.png'))

def create_simplified_graph(G, seeds=None, highlight=None, out='simple_graph.png'):
    seeds = seeds or []
    highlight = highlight or []
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='lightblue', alpha=0.6)
    
    if seeds:
        seed_nodes_in_graph = [n for n in seeds if n in G]
        nx.draw_networkx_nodes(G, pos, nodelist=seed_nodes_in_graph,
                             node_size=50, node_color='red', alpha=1.0)
    
    if highlight:
        highlight_nodes_in_graph = [n for n in highlight if n in G]
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes_in_graph,
                             node_size=30, node_color='green', alpha=1.0)
    
    plt.axis('off')
    plt.title(f'Simplified Network View\n{G.number_of_nodes()} nodes', fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'[viz] Saved simplified view: {out}')

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_pipeline(maildir: str, k_seeds: int = 5, seed_method: str = 'degree_discount', 
                p: float = 0.1, plot=True, viz=False, viz_nodes: int = 500, 
                layout: str = 'spring', spring_k=None, comm=False, fast=False, 
                eig=False, figsize=(16, 12)):
    emails = parse_emails(maildir)
    print(f'Emails: {len(emails):,}')
    
    G = build_graph(emails)
    print(f'Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')
    
    sens = [e for e in emails if is_sensitive(e['body'])]
    ph = [e for e in emails if is_phishing(e['body'])]
    print(f'Sensitive: {len(sens):,}  Phishing: {len(ph):,}')
    
    anomalies = detect_anomalies(G, emails)
    if anomalies:
        print('Anomalies:', anomalies)
    
    if fast:
        seed_method = 'degree_discount'
    
    seeds = select_seeds(G, k=k_seeds, method=seed_method, eig=eig)
    print('Seeds:', seeds)
    
    cas = independent_cascade(G, seeds, p)
    print(f'Cascade reached {len(cas.active):,} nodes')
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(cas.timeline)), cas.timeline, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Infected Nodes', fontsize=12)
        plt.title('Information Cascade Timeline', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cascade_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[plot] Saved cascade timeline: cascade_plot.png')
    
    if viz:
        infected = [r.strip() for e in ph for r in e['to'].split(',') if r.strip()]
        draw_graph(G, seeds=seeds, highlight=infected, max_nodes=viz_nodes, 
                  out='email_graph.png', layout=layout, spring_k=spring_k, 
                  colour_comm=comm, figsize=figsize)
    
    inf_nodes = [r.strip() for e in ph for r in e['to'].split(',') if r.strip()]
    if inf_nodes:
        origins = trace_sources(G, inf_nodes)
        print('Likely origins:', origins)
    
    # Generate preventive measures
    generate_preventive_measures(emails, anomalies, ph, inf_nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Phishing Simulation Pipeline with Preventive Measures")
    parser.add_argument("maildir", type=str, help="Path to email directory")
    parser.add_argument("--viz", action="store_true", help="Generate graph visualization")
    parser.add_argument("--viz-nodes", type=int, default=500, help="Number of nodes to draw")
    parser.add_argument("--layout", type=str, choices=["spring", "kk", "circular", "shell", "spectral"], 
                       default="spring", help="Graph layout algorithm")
    parser.add_argument("--spring-k", type=float, help="Spring layout k value (distance between nodes)")
    parser.add_argument("--comm", action="store_true", help="Color nodes by community")
    parser.add_argument("--fast", action="store_true", help="Use degree_discount for faster execution")
    parser.add_argument("--eig", action="store_true", help="Include eigenvector centrality in composite")
    parser.add_argument("--seed-method", type=str, default="composite", 
                       choices=["degree_discount", "degree", "composite", "random"],
                       help="Seed selection method")
    parser.add_argument("--k-seeds", type=int, default=5, help="Number of seed nodes")
    parser.add_argument("--p", type=float, default=0.1, help="Infection probability")
    parser.add_argument("--figsize", type=str, default="16,12", 
                       help="Figure size as 'width,height' (e.g., '16,12')")
    
    args = parser.parse_args()
    
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except:
        print("Warning: Invalid figsize format, using default (16,12)")
        figsize = (16, 12)
    
    run_pipeline(
        maildir=args.maildir,
        k_seeds=args.k_seeds,
        seed_method=args.seed_method,
        p=args.p,
        plot=True,
        viz=args.viz,
        viz_nodes=args.viz_nodes,
        layout=args.layout,
        spring_k=args.spring_k,
        comm=args.comm,
        fast=args.fast,
        eig=args.eig,
        figsize=figsize
    )
