"""Small speaker models trained from conservative development-only anchors."""
import numpy as np
from collections import defaultdict, Counter
from rapidfuzz.fuzz import ratio
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def normalize_vector(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)

def page_clusters(page):
    groups = defaultdict(list)
    for c, label in enumerate(page['character_cluster_labels']):
        groups[int(label)].append(c)
    return groups

def transcript_anchors(page, rows, normalizer):
    # A repeated phrase with competing speakers is not a reliable identity anchor.
    anchors = {}
    for i, raw in enumerate(page['ocr']):
        text = normalizer(raw)
        if len(text) < 8:
            continue
        scores = sorted([(ratio(text, normalizer(r['text']))/100, r['speaker'])
                         for r in rows], key=lambda x: x[0], reverse=True)
        if not scores or scores[0][0] < 0.94:
            continue
        best, label = scores[0]
        if label in (None, 'UNKNOWN', 'NARRATION'):
            continue
        if any(speaker != label and score >= best-0.05 for score, speaker in scores[1:]):
            continue
        anchors[i] = label
    return anchors

def anchor_votes(page, anchors):
    votes = defaultdict(list)
    for t, c in page['text_character_associations']:
        if t in anchors:
            votes[int(page['character_cluster_labels'][c])].append((t, anchors[t]))
    return votes

def agreed_identity(votes, exclude=None):
    labels = [label for text, label in votes if text != exclude]
    # Two independent balloons must agree. Contradictory clusters are left unlabelled.
    return labels[0] if len(labels) >= 2 and len(set(labels)) == 1 else None

def cluster_nodes(pages):
    nodes, appearances = [], {}
    for p, page in enumerate(pages):
        for label, members in page_clusters(page).items():
            prototype = normalize_vector(np.mean([
                normalize_vector(page['char_embeddings'][c]) for c in members], axis=0))
            index = len(nodes)
            nodes.append({'page': p, 'local': label, 'embedding': prototype})
            for c in members:
                appearances[p, c] = index
    return nodes, appearances

def panel_for(box, panels):
    x1, y1, x2, y2 = box
    area = max((x2-x1)*(y2-y1), 1.)
    overlap = [max(0., min(x2,b[2])-max(x1,b[0])) * max(0., min(y2,b[3])-max(y1,b[1])) / area for b in panels]
    return int(np.argmax(overlap)) if overlap and max(overlap) >= 0.3 else -1

def box_info(box, size):
    w, h = size
    x1, y1, x2, y2 = np.asarray(box) / np.array([w,h,w,h])
    return np.array([(x1+x2)/2, (y1+y2)/2, max(x2-x1,0), max(y2-y1,0)])

def link_features(page, text, char):
    a = box_info(page['texts'][text], page['size'])
    b = box_info(page['characters'][char], page['size'])
    delta = b[:2]-a[:2]
    tp, cp = panel_for(page['texts'][text],page['panels']), panel_for(page['characters'][char],page['panels'])
    links = {int(t): int(c) for t,c in page['text_character_associations']}
    old = links.get(text)
    labels = page['character_cluster_labels']
    tails = {int(t):int(tail) for t,tail in page.get('text_tail_associations',[])}
    tail = tails.get(text)
    td = box_info(page['tails'][tail],page['size'])[:2]-b[:2] if tail is not None else np.zeros(2)
    same_old_cluster = old is not None and labels[old] == labels[char]
    # Tail boxes provide proximity, not an inferred tail direction.
    return np.r_[a,b,delta,np.abs(delta),np.linalg.norm(delta),
                 tp >= 0 and tp == cp, tp < 0 or cp < 0,
                 old == char, same_old_cluster, tail is not None,
                 td,np.linalg.norm(td),
                 sum(label == labels[char] for label in labels)/max(len(labels),1)].astype(np.float32)

def pair_features(a, b, projector):
    za, zb = projector.transform(np.stack([a,b]))
    return np.r_[float(a@b), np.abs(za-zb), za*zb].astype(np.float32)

def fit_speaker_models(ids, cache, gold, normalizer):
    pair_records, link_x, link_y, link_weights, audit = [], [], [], [], []
    for sid in ids:
        pages = cache[sid]
        nodes, _ = cluster_nodes(pages)
        identity = {}
        pair_start = len(pair_records)
        for p, page in enumerate(pages):
            anchors = transcript_anchors(page, gold[sid][p], normalizer)
            votes = anchor_votes(page, anchors)
            for local, entries in votes.items():
                label = agreed_identity(entries)
                if label is not None:
                    identity[p, local] = label
            for t, speaker in anchors.items():
                candidates = []
                for c, local in enumerate(page['character_cluster_labels']):
                    # The current balloon must not manufacture its own training target.
                    label = agreed_identity(votes[int(local)], exclude=t)
                    if label is not None:
                        candidates.append((c, int(label == speaker)))
                if {y for _,y in candidates} != {0,1}:
                    continue
                counts = Counter(y for _,y in candidates)
                for c, target in candidates:
                    link_x.append(link_features(page,t,c));link_y.append(target)
                    link_weights.append(1/counts[target])
                    audit.append({'sequence_id':sid,'page':p+1,'text_index':t,
                                  'character_index':c,'same_speaker':target,
                                  'target_source':'other agreeing balloons in local cluster'})
        for i,a in enumerate(nodes):
            la=identity.get((a['page'],a['local']))
            if la is None:
                continue
            for b in nodes[:i]:
                lb=identity.get((b['page'],b['local']))
                if lb is not None and a['page'] != b['page']:
                    pair_records.append((a['embedding'],b['embedding'],int(la==lb),sid))
    if not pair_records or len({r[2] for r in pair_records}) < 2:
        raise RuntimeError('Not enough cross-page anchor pairs. Annotate development examples instead of inventing labels.')
    if len(set(link_y)) < 2:
        raise RuntimeError('Not enough independently anchored balloon candidates for ranker training.')
    embeddings=np.stack([e for a,b,_,_ in pair_records for e in (a,b)])
    projector=PCA(n_components=min(24,len(embeddings)-1,embeddings.shape[1]),random_state=42).fit(embeddings)
    px=np.stack([pair_features(a,b,projector) for a,b,_,_ in pair_records]);py=np.array([y for _,_,y,_ in pair_records])
    pair_counts=Counter(sid for *_,sid in pair_records)
    pw=np.array([1/pair_counts[sid] for *_,sid in pair_records]);pw*=len(pw)/pw.sum()
    def classifier(C):
        return make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=42))
    matcher=classifier(.05).fit(px,py,logisticregression__sample_weight=pw)
    ranker=classifier(.1).fit(np.stack(link_x),link_y,logisticregression__sample_weight=link_weights)
    return {'projector':projector,'matcher':matcher,'ranker':ranker}, {
        'pair_counts':dict(Counter(map(int,py))), 'ranker_counts':dict(Counter(map(int,link_y))),
        'pair_sequences':len(pair_counts),'ranker_sequences':len({r['sequence_id'] for r in audit}),
        'ranker_training_balloons':len({(r['sequence_id'],r['page'],r['text_index']) for r in audit}),
        'supervision':'weak labels from agreeing OCR-aligned development balloons; no box-level ground truth'}, audit

def prepare_speaker_predictions(pages, models):
    nodes, appearances=cluster_nodes(pages)
    n=len(nodes); probabilities=np.eye(n);cosine=np.eye(n)
    pairs=[(i,j) for i,a in enumerate(nodes) for j,b in enumerate(nodes[:i]) if a['page']!=b['page']]
    if pairs:
        features=np.stack([pair_features(nodes[i]['embedding'],nodes[j]['embedding'],models['projector']) for i,j in pairs])
        scores=models['matcher'].predict_proba(features)[:,1]
        for (i,j),score in zip(pairs,scores):
            probabilities[i,j]=probabilities[j,i]=score
            cosine[i,j]=cosine[j,i]=float(nodes[i]['embedding']@nodes[j]['embedding'])
    ranking=[]
    for page in pages:
        nt,nc=len(page['ocr']),len(page['characters'])
        scores=np.zeros((nt,nc))
        if nt and nc:
            scores=models['ranker'].predict_proba(np.stack([
                link_features(page,t,c) for t in range(nt) for c in range(nc)]))[:,1].reshape(nt,nc)
        ranking.append(scores)
    return {'nodes':nodes,'appearances':appearances,'learned':probabilities,'cosine':cosine,'ranking':ranking}

def merge_prepared(prepared, learned, threshold):
    nodes=prepared['nodes'];matrix=prepared['learned' if learned else 'cosine']
    groups=[{i} for i in range(len(nodes))]
    edges=sorted([(matrix[i,j],i,j) for i,a in enumerate(nodes) for j,b in enumerate(nodes[:i]) if a['page']!=b['page']],reverse=True)
    for score,i,j in edges:
        if score<threshold: break
        a=next(g for g in groups if i in g);b=next(g for g in groups if j in g)
        if a is b or {nodes[k]['page'] for k in a}&{nodes[k]['page'] for k in b}: continue
        if min(matrix[x,y] for x in a for y in b)<threshold: continue
        groups=[g for g in groups if g is not a and g is not b]+[a|b]
    ids={n:i for i,g in enumerate(sorted(groups,key=min)) for n in g}
    return {key:ids[value] for key,value in prepared['appearances'].items()}

def upgraded_prediction(pages, prepared, cfg):
    identities=merge_prepared(prepared,cfg['learned_matcher'],cfg['match_threshold'])
    names={};output=[]
    for p,page in enumerate(pages):
        original={int(t):int(c) for t,c in page['text_character_associations']}
        rows=[]
        for t,raw in enumerate(page['ocr']):
            text=' '.join(str(raw).split())
            if not text or not page['is_essential_text'][t]: continue
            if page['narr_prob'][t]>=.85:
                speaker='NARRATION'
            else:
                chosen=original.get(t)
                if cfg['use_ranker'] and len(page['characters']):
                    scores=prepared['ranking'][p][t]
                    # Compare cluster scores, not two crops of the same character.
                    cluster_best={}
                    for c,local in enumerate(page['character_cluster_labels']):
                        if local not in cluster_best or scores[c]>cluster_best[local][0]:
                            cluster_best[local]=(scores[c],c)
                    ranked=sorted(cluster_best.values(),reverse=True)
                    best,candidate=ranked[0];runner=ranked[1][0] if len(ranked)>1 else 0.
                    if best>=cfg['link_threshold'] and best-runner>=cfg['link_margin']:
                        chosen=candidate
                key=('character',identities[p,chosen]) if chosen is not None else ('unresolved',p,t)
                if key not in names: names[key]=f'char{len(names)+1}'
                speaker=names[key]
            rows.append({'speaker':speaker,'text':text})
        output.append(rows)
    return output
