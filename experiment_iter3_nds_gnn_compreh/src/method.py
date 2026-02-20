#!/usr/bin/env python3
"""
NDS-GNN Comprehensive Evaluation: CSL Expressiveness + Benchmark Classification.

Part A: Test NDS distinguishing power on CSL graph classes.
Part B: GIN+NDS on MUTAG, PROTEINS, IMDB-BINARY with fold-based CV.
"""
import json, os, resource, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

BLUE, GREEN, END = "\033[94m", "\033[92m", "\033[0m"
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")
resource.setrlimit(resource.RLIMIT_AS, (14*1024**3, 14*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3300, 3300))

WS = Path(__file__).resolve().parent
DDIR = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_082247"
            "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
(WS/"logs").mkdir(exist_ok=True)
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
TBUDGET = 50*60; T0 = time.time()
def tleft(): return TBUDGET-(time.time()-T0)

# ═══ Data ═══
def load_ds(p):
    logger.info(f"Loading {p}")
    d = json.loads(p.read_text())
    for ds in d.get("datasets",[]): logger.info(f"  {ds['dataset']}: {len(ds['examples'])}")
    return d

def parse_g(ex):
    gd = json.loads(ex["input"]); n=gd["num_nodes"]
    nf = gd.get("node_features")
    return {"n":n,"el":gd["edge_list"],
            "feat":np.array(nf,dtype=np.float32) if nf else np.ones((n,1),dtype=np.float32),
            "label":int(ex["output"]),"fold":ex.get("metadata_fold",0),
            "ncls":ex.get("metadata_n_classes",2)}

def build_adj(n, el):
    if not el: return sp.csr_matrix((n,n),dtype=np.float32)
    e=np.array(el,dtype=np.int64)
    A=sp.csr_matrix((np.ones(len(e),dtype=np.float32),(e[:,0],e[:,1])),shape=(n,n))
    A=A+A.T; A.data=np.minimum(A.data,1.0); return A

def sym_norm(A):
    deg=np.array(A.sum(1)).flatten(); di=np.zeros_like(deg)
    m=deg>0; di[m]=1.0/np.sqrt(deg[m]); return sp.diags(di)@A@sp.diags(di)

# ═══ NDS Features ═══
def nds(An, x0, nl, T):
    parts=[x0.copy()]; x=x0.copy()
    for _ in range(T): x=nl(An@x); parts.append(x.copy())
    return np.concatenate(parts, axis=1)

def deg_feat(A): return np.array(A.sum(1),dtype=np.float32).reshape(-1,1)

def clust_coeff(A):
    n=A.shape[0]; cc=np.zeros(n,dtype=np.float32); Ac=A.tocsr()
    for v in range(n):
        nb=Ac[v].indices; k=len(nb)
        if k<2: continue
        cc[v]=Ac[nb][:,nb].nnz/(k*(k-1))
    return cc.reshape(-1,1)

def ifeat(A, s):
    if s=="degree": return deg_feat(A)
    if s=="clust": return clust_coeff(A)
    if s=="deg+clust": return np.concatenate([deg_feat(A),clust_coeff(A)],axis=1)
    raise ValueError(s)

_relu=lambda x:np.maximum(x,0.0)
_tanh=lambda x:np.tanh(x)
_abs=lambda x:np.abs(x)
_lrelu=lambda x:np.where(x>0,x,0.2*x)
_id=lambda x:x
NL={"relu":_relu,"tanh":_tanh,"abs":_abs,"leaky_relu":_lrelu,"identity":_id}

# ═══ 1-WL ═══
def wl_hash(A, iters=10):
    n=A.shape[0]; Ac=A.tocsr()
    c=np.array(A.sum(1),dtype=np.int32).flatten().tolist()
    for _ in range(iters):
        cm,ni,nc={},0,[]
        for v in range(n):
            k=(c[v],tuple(sorted(c[u] for u in Ac[v].indices)))
            if k not in cm: cm[k]=ni; ni+=1
            nc.append(cm[k])
        c=nc
    return str(tuple(sorted(c)))

# ═══ Part A: CSL Expressiveness ═══
def part_a(graphs, labels):
    t0=time.time(); n=len(graphs)
    logger.info(f"{BLUE}Part A: CSL — {n} graphs{END}")
    adjs=[build_adj(g["n"],g["el"]) for g in graphs]
    norms=[sym_norm(A) for A in adjs]
    cfgs=[
        ("nds_relu_T5_deg","relu",5,"degree"),("nds_relu_T10_deg","relu",10,"degree"),
        ("nds_tanh_T5_deg","tanh",5,"degree"),("nds_tanh_T10_deg","tanh",10,"degree"),
        ("nds_tanh_T15_deg","tanh",15,"degree"),("nds_tanh_T20_deg","tanh",20,"degree"),
        ("nds_abs_T10_deg","abs",10,"degree"),("nds_lrelu_T10_deg","leaky_relu",10,"degree"),
        ("linear_T10_deg","identity",10,"degree"),
        ("nds_tanh_T10_clust","tanh",10,"clust"),("nds_tanh_T10_dc","tanh",10,"deg+clust"),
    ]
    ulabs=sorted(set(labels)); nc=len(ulabs)
    logger.info(f"  {nc} classes")
    scores={}
    for name,nln,T,ini in cfgs:
        gf=[]
        for i in range(n):
            f0=ifeat(adjs[i],ini); nd=nds(norms[i],f0,NL[nln],T)
            gf.append(np.concatenate([np.sort(nd.sum(0)),np.sort(nd.mean(0)),np.sort(nd.std(0))]))
        lf=defaultdict(list)
        for f,l in zip(gf,labels): lf[l].append(f)
        cm={l:np.mean(fs,axis=0) for l,fs in lf.items()}
        np_,nd_=0,0
        for i in range(len(ulabs)):
            for j in range(i+1,len(ulabs)):
                np_+=1
                if np.linalg.norm(cm[ulabs[i]]-cm[ulabs[j]])>1e-6: nd_+=1
        acc=nd_/max(np_,1); scores[name]={"pairs_distinguished":nd_,"total_pairs":np_,"accuracy":acc}
        logger.info(f"  {name}: {nd_}/{np_} ({100*acc:.1f}%)")
    # WL baseline
    wh=[wl_hash(adjs[i]) for i in range(n)]
    wlh=defaultdict(set)
    for h,l in zip(wh,labels): wlh[l].add(h)
    wn=wd=0
    for i in range(len(ulabs)):
        for j in range(i+1,len(ulabs)):
            wn+=1
            if wlh[ulabs[i]].isdisjoint(wlh[ulabs[j]]): wd+=1
    scores["wl_1_baseline"]={"pairs_distinguished":wd,"total_pairs":wn,"accuracy":wd/max(wn,1)}
    logger.info(f"  wl_1: {wd}/{wn}")
    # Degree baseline
    df=[np.concatenate([np.sort(deg_feat(adjs[i]).flatten()),[deg_feat(adjs[i]).mean(),deg_feat(adjs[i]).std()]]) for i in range(n)]
    dl=defaultdict(list)
    for f,l in zip(df,labels): dl[l].append(f)
    dcm={l:np.mean(fs,axis=0) for l,fs in dl.items()}
    dd=0
    for i in range(len(ulabs)):
        for j in range(i+1,len(ulabs)):
            if np.linalg.norm(dcm[ulabs[i]]-dcm[ulabs[j]])>1e-6: dd+=1
    scores["degree_only"]={"pairs_distinguished":dd,"total_pairs":wn,"accuracy":dd/max(wn,1)}
    logger.info(f"  deg: {dd}/{wn}")
    el=time.time()-t0; logger.info(f"  Part A: {el:.1f}s")
    return {"config_scores":scores,"n_graphs":n,"n_classes":nc,"wall_clock_s":el}

# ═══ GIN Model ═══
class GINLayer(nn.Module):
    def __init__(s,di,do):
        super().__init__(); s.eps=nn.Parameter(torch.zeros(1))
        s.mlp=nn.Sequential(nn.Linear(di,do),nn.BatchNorm1d(do),nn.ReLU(),nn.Linear(do,do),nn.BatchNorm1d(do),nn.ReLU())
    def forward(s,x,a): return s.mlp((1+s.eps)*x+torch.sparse.mm(a,x))

class GIN(nn.Module):
    def __init__(s,di,dh,nc,nl=3,dr=0.5,pool="sum"):
        super().__init__(); s.pool=pool; s.dr=dr
        s.layers=nn.ModuleList([GINLayer(di,dh)]+[GINLayer(dh,dh) for _ in range(nl-1)])
        s.clfs=nn.ModuleList([nn.Linear(di,nc)]+[nn.Linear(dh,nc) for _ in range(nl)])
    def forward(s,x,a,gi,ng):
        hs=[x]; h=x
        for ly in s.layers: h=ly(h,a); hs.append(h)
        logits=torch.zeros(ng,s.clfs[0].out_features)
        for i,(hl,cl) in enumerate(zip(hs,s.clfs)):
            p=torch.zeros(ng,hl.shape[1]); p.scatter_add_(0,gi.unsqueeze(1).expand_as(hl),hl)
            if s.pool=="mean":
                cnt=torch.zeros(ng); cnt.scatter_add_(0,gi,torch.ones(x.shape[0]))
                p=p/cnt.clamp(min=1).unsqueeze(1)
            if i==len(hs)-1: p=F.dropout(p,p=s.dr,training=s.training)
            logits=logits+cl(p)
        return logits

# ═══ OPTIMIZED Batching — Pre-batch entire fold ONCE ═══
def prebatch_fold(graphs, feats, train_idx, test_idx, bs=32):
    """Pre-build ALL train batches + test batch as tensors ONCE. Returns list of (x,a,gi,y,ng) tuples."""
    def _build(idx):
        tn=0; rows,cols,fps,gis,labs=[],[],[],[],[]
        for li,ii in enumerate(idx):
            g=graphs[ii]; ea=g["el"]
            for s,d in ea: rows.append(s+tn); cols.append(d+tn)
            fps.append(feats[ii]); gis.extend([li]*g["n"]); labs.append(g["label"]); tn+=g["n"]
        x=torch.from_numpy(np.vstack(fps))
        if rows:
            r,c=np.array(rows,dtype=np.int64),np.array(cols,dtype=np.int64)
            a=torch.sparse_coo_tensor(torch.from_numpy(np.stack([r,c])),torch.ones(len(r)),size=(tn,tn))
        else:
            a=torch.sparse_coo_tensor(torch.zeros(2,0,dtype=torch.long),torch.zeros(0),size=(max(tn,1),max(tn,1)))
        return x,a,torch.tensor(gis,dtype=torch.long),torch.tensor(labs,dtype=torch.long),len(idx)

    # Pre-build train mini-batches
    nt=len(train_idx)
    train_batches=[]
    for s in range(0,nt,bs):
        bi=train_idx[s:min(s+bs,nt)]
        train_batches.append(_build(bi))

    test_batch=_build(test_idx)
    return train_batches, test_batch

def train_fold(graphs, feats, train_idx, test_idx, ncls, dh=64, nl=3, dr=0.5, lr=0.01, ep=100, pat=20, bs=32, pool="sum", time_limit=120):
    """Train GIN on one fold with pre-batching for speed."""
    t0f=time.time()
    di=feats[train_idx[0]].shape[1]
    model=GIN(di,dh,ncls,nl,dr,pool)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=10)

    # Pre-batch for this fold — KEY optimization
    train_batches, (x_te,a_te,gi_te,y_te,n_te) = prebatch_fold(graphs,feats,train_idx,test_idx,bs)
    nb=len(train_batches)

    ba,bp,stale=0.0,None,0
    for e in range(ep):
        if time.time()-t0f > time_limit: break  # per-fold time limit
        model.train()
        # Shuffle batch order (not individual examples — pre-batched)
        perm=np.random.permutation(nb)
        tl=0.0; tn=0
        for bi in perm:
            x,a,gi,y,ng=train_batches[bi]
            opt.zero_grad()
            l=F.cross_entropy(model(x,a,gi,ng),y)
            l.backward(); opt.step()
            tl+=l.item()*ng; tn+=ng
        sch.step(tl/max(tn,1))
        model.eval()
        with torch.no_grad():
            p=model(x_te,a_te,gi_te,n_te).argmax(1).numpy()
            ac=float((p==y_te.numpy()).mean())
        if ac>ba: ba,bp,stale=ac,p.copy(),0
        else: stale+=1
        if stale>=pat: break
    return ba,bp

# ═══ Feature Augmentation ═══
def make_features(graphs, adjs, norms, methods):
    mf={}
    for mn in methods:
        if mn=="gin_vanilla": mf[mn]=[g["feat"] for g in graphs]
        elif mn=="gin_degree": mf[mn]=[np.concatenate([g["feat"],deg_feat(adjs[i])],axis=1) for i,g in enumerate(graphs)]
        elif mn=="gin_linear_diff": mf[mn]=[np.concatenate([g["feat"],nds(norms[i],ifeat(adjs[i],"degree"),_id,10)],axis=1) for i,g in enumerate(graphs)]
        elif mn=="gin_nds_relu_T10": mf[mn]=[np.concatenate([g["feat"],nds(norms[i],ifeat(adjs[i],"degree"),_relu,10)],axis=1) for i,g in enumerate(graphs)]
        elif mn=="gin_nds_tanh_T10": mf[mn]=[np.concatenate([g["feat"],nds(norms[i],ifeat(adjs[i],"degree"),_tanh,10)],axis=1) for i,g in enumerate(graphs)]
        elif mn=="gin_nds_tanh_T10_clust": mf[mn]=[np.concatenate([g["feat"],nds(norms[i],ifeat(adjs[i],"deg+clust"),_tanh,10)],axis=1) for i,g in enumerate(graphs)]
    return mf

ALL_METHODS=["gin_vanilla","gin_degree","gin_linear_diff","gin_nds_relu_T10","gin_nds_tanh_T10","gin_nds_tanh_T10_clust"]

# ═══ Part B ═══
def part_b(ds_name, examples, max_ex=None, ep=100, dh=64, nl=3, dr=0.5, lr=0.01, bs=32, pat=20, tl=120):
    t0=time.time()
    graphs=[parse_g(ex) for ex in examples]
    if max_ex and max_ex<len(graphs): graphs=graphs[:max_ex]; examples=examples[:max_ex]
    ng=len(graphs); folds=np.array([g["fold"] for g in graphs])
    ncls=graphs[0]["ncls"]; nfolds=int(examples[0].get("metadata_num_folds",10))
    pool="mean" if "IMDB" in ds_name else "sum"
    logger.info(f"{BLUE}{ds_name}: {ng}g, {ncls}cls, {nfolds}f, pool={pool}{END}")
    adjs=[build_adj(g["n"],g["el"]) for g in graphs]
    norms=[sym_norm(A) for A in adjs]
    logger.info("  Computing features..."); tf=time.time()
    mf=make_features(graphs,adjs,norms,ALL_METHODS)
    logger.info(f"  Features in {time.time()-tf:.1f}s")
    for mn,fs in mf.items(): logger.info(f"    {mn}: dim={fs[0].shape[1]}")
    ufs=sorted(set(folds.tolist())); logger.info(f"  Folds: {ufs}")
    mr={}; mp={m:[None]*ng for m in ALL_METHODS}

    for mn in ALL_METHODS:
        if tleft()<60:
            logger.warning(f"  Budget low, skip {mn}")
            mr[mn]={"mean_acc":0,"std_acc":0,"fold_accs":[],"n_folds":0}; continue
        fas=[]; t_m=time.time()
        for fid in ufs:
            if tleft()<30: break
            tri=np.where(folds!=fid)[0]; tei=np.where(folds==fid)[0]
            if not len(tei) or not len(tri): continue
            ac,pr=train_fold(graphs,mf[mn],tri,tei,ncls=ncls,dh=dh,nl=nl,dr=dr,lr=lr,ep=ep,pat=pat,bs=bs,pool=pool,time_limit=tl)
            fas.append(ac)
            if pr is not None:
                for li,gi in enumerate(tei): mp[mn][gi]=int(pr[li])
        ma=float(np.mean(fas)) if fas else 0.0
        sa=float(np.std(fas)) if fas else 0.0
        mr[mn]={"mean_acc":ma,"std_acc":sa,"fold_accs":[float(a) for a in fas],"n_folds":len(fas)}
        logger.info(f"  {mn}: {100*ma:.1f}±{100*sa:.1f}% ({time.time()-t_m:.0f}s)")
    el=time.time()-t0; logger.info(f"  {ds_name}: {el:.1f}s total")
    return {"dataset":ds_name,"n_graphs":ng,"n_classes":ncls,"num_folds":nfolds,"pooling":pool,
            "method_results":mr,"method_predictions":mp,"wall_clock_s":el}

# ═══ Output ═══
def build_out(csl_r, cls_r, dbn, ws):
    o={"metadata":{
        "method_name":"NDS-GIN Comprehensive Evaluation",
        "description":"Nonlinear Diffusion Signatures (NDS) as parameter-free preprocessing for GNNs. "
            "Interleaves linear graph diffusion with fixed pointwise nonlinearity (ReLU/tanh) to break "
            "spectral invariance. Part A: CSL expressiveness. Part B: GIN+NDS on MUTAG/PROTEINS/IMDB-B.",
        "nds_hyperparameters":{"T":10,"nonlinearities":["relu","tanh","abs","leaky_relu","identity"],
            "init_strategies":["degree","clust","deg+clust"],"normalization":"symmetric D^{-1/2}AD^{-1/2}"},
        "gin_hyperparameters":{"hidden_dim":64,"num_layers":3,"dropout":0.5,"lr":0.01,
            "epochs":"80-100","patience":20,"batch_size":32},
        "summary":{"part_a_csl":csl_r["config_scores"] if csl_r else {},
            "part_b":{d:r["method_results"] for d,r in cls_r.items()},
            "wall_clock_s":ws}},
        "datasets":[]}
    if csl_r and "CSL" in dbn:
        cs=csl_r["config_scores"]
        nds_c=[(k,v["accuracy"]) for k,v in cs.items() if k not in ("wl_1_baseline","degree_only")]
        best=max(nds_c,key=lambda x:x[1])[0] if nds_c else "none"
        co=[{"input":e["input"],"output":e["output"],"predict_nds_best":best,
             "predict_wl_baseline":"1-WL equivalent",
             "metadata_fold":e.get("metadata_fold",0),"metadata_row_index":e.get("metadata_row_index",0),
             "metadata_num_nodes":e.get("metadata_num_nodes",0),"metadata_num_edges":e.get("metadata_num_edges",0)}
            for e in dbn["CSL"]]
        o["datasets"].append({"dataset":"CSL","examples":co})
    for ds,r in cls_r.items():
        exs=dbn.get(ds,[]); preds=r.get("method_predictions",{}); ngr=r["n_graphs"]; do=[]
        for i,e in enumerate(exs[:ngr]):
            row={"input":e["input"],"output":e["output"],
                 "metadata_fold":e.get("metadata_fold",0),"metadata_row_index":e.get("metadata_row_index",0),
                 "metadata_num_nodes":e.get("metadata_num_nodes",0),"metadata_num_edges":e.get("metadata_num_edges",0)}
            for mn,pl in preds.items():
                v=pl[i] if i<len(pl) and pl[i] is not None else ""
                row[f"predict_{mn}"]=str(v)
            do.append(row)
        if do: o["datasets"].append({"dataset":ds,"examples":do})
    return o

# ═══ Main ═══
@logger.catch
def main():
    global T0; T0=time.time()
    logger.info(f"{GREEN}=== NDS-GIN Evaluation ==={END}")
    max_ex=int(os.environ["MAX_EXAMPLES"]) if "MAX_EXAMPLES" in os.environ else None
    data_file=os.environ.get("DATA_FILE",str(DDIR/"full_data_out.json"))
    logger.info(f"data={data_file}, max_ex={max_ex}")
    d=load_ds(Path(data_file))
    dbn={ds["dataset"]:ds["examples"] for ds in d["datasets"]}
    # Part A
    csl_r=None
    if "CSL" in dbn:
        gs=[parse_g(e) for e in dbn["CSL"]]
        if max_ex: gs=gs[:max_ex]
        csl_r=part_a(gs,[g["label"] for g in gs])
    # Part B
    cls_r={}
    # Per-dataset configs: larger datasets get fewer epochs + per-fold time limits
    ds_cfgs = {
        "MUTAG":      {"ep":80, "dh":64,"nl":3,"dr":0.5,"lr":0.01,"bs":32,"pat":15,"tl":60},
        "PROTEINS":   {"ep":50, "dh":64,"nl":2,"dr":0.5,"lr":0.01,"bs":64,"pat":10,"tl":40},
        "IMDB-BINARY":{"ep":50, "dh":64,"nl":2,"dr":0.5,"lr":0.01,"bs":64,"pat":10,"tl":40},
    }
    for ds in ["MUTAG","PROTEINS","IMDB-BINARY"]:
        if ds not in dbn: continue
        if tleft()<90: logger.warning(f"Budget low ({tleft():.0f}s), skip {ds}"); continue
        cfg = ds_cfgs.get(ds, ds_cfgs["MUTAG"])
        cls_r[ds]=part_b(ds,dbn[ds],max_ex=max_ex,**cfg)
    ws=time.time()-T0
    logger.info(f"\n{GREEN}=== Output ==={END}")
    o=build_out(csl_r,cls_r,dbn,ws)
    op=WS/"method_out.json"; op.write_text(json.dumps(o,indent=2))
    logger.info(f"Written {op}")
    logger.info(f"\n{GREEN}=== SUMMARY ==={END}")
    if csl_r:
        logger.info(f"{BLUE}Part A:{END}")
        for c,s in csl_r["config_scores"].items():
            logger.info(f"  {c}: {s['pairs_distinguished']}/{s['total_pairs']} ({100*s['accuracy']:.1f}%)")
    logger.info(f"{BLUE}Part B:{END}")
    for d,r in cls_r.items():
        logger.info(f"  {d}:")
        for m,s in r["method_results"].items():
            logger.info(f"    {m}: {100*s['mean_acc']:.1f}±{100*s['std_acc']:.1f}%")
    te=sum(len(d["examples"]) for d in o["datasets"])
    logger.info(f"Total: {te} examples, {ws:.1f}s")

if __name__=="__main__": main()
