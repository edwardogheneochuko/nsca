import os
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from load_data import load
from encoder import SymptomEncoder
from esoinn import TwoLayerESOINN
from explain import Explainer
from srm import SRM

app = FastAPI(title="NSCA disease identification")
S = {}
nice = lambda s: s.replace("_", " ")

@app.on_event("startup")
def boot():
    X, y, desc, prec = load(); A = X.values.astype(float)
    enc = SymptomEncoder(X.columns).fit_idf(A)
    net = TwoLayerESOINN(enc.omega).fit(A)                      # Stage 1
    ex = Explainer(net, A, X.columns)                           # Stage 2
    srm = SRM(A, y, X.columns, enc.omega, desc, prec)           # Stage 3
    for k in range(ex.K): srm.ground_cluster(k, ex.centers[k])
    S.update(enc=enc, net=net, ex=ex, srm=srm, vocab=list(X.columns))

class Req(BaseModel):
    symptoms: list[str]
    absent: list[str] = []

@app.get("/api/symptoms")
def symptoms(): return [dict(id=s, label=nice(s)) for s in S["vocab"]]

@app.get("/api/info")
def info():
    n, ex = S["net"], S["ex"]
    return dict(l1_nodes=len(n.l1.w), l2_nodes=len(n.l2.w), clusters=ex.K, tree_fidelity=ex.fidelity,
                cluster_labels={k: S["srm"].ground_cluster(k, ex.centers[k]) for k in range(ex.K)})

@app.post("/api/diagnose")
def diagnose(r: Req):
    enc, net, ex, srm = S["enc"], S["net"], S["ex"], S["srm"]
    x, m = enc.encode(r.symptoms, r.absent)
    mu = ex.mu(x)[0]; order = np.argsort(-mu); k = int(order[0])
    n1, d1 = net.l1.winner(x, m)
    path, leaf = ex.path(x); shap_l = ex.shap_local(x, k)
    cfs = [ex.counterfactual(x, k, int(t)) for t in order[1:3]]
    g = srm.diagnose(x, k, path)
    for c in cfs:
        c["leading_disease"] = srm.dis[srm.cands[c["to"]][0]]
        c["question"] = "; ".join(f"{'Do you have' if a == 'add' else 'Is it true you do NOT have'} {nice(s)}?" for s, a in c["changes"])
    if len([s for s in r.symptoms if s in S["vocab"]]) < 3: g["status"] = "insufficient"
    entered = [nice(s) for s in r.symptoms if s in S["vocab"]]
    stages = [
        dict(stage="Stage 0 - Encoding", text=f"{len(entered)} symptoms mapped to a {len(S['vocab'])}-slot vector with IDF weights (rare symptoms count more)."),
        dict(stage="Stage 1 - ESOINN", text=f"Closest typical presentation: L1 node {n1} (distance {d1:.2f}). Closest disease cluster: L2 cluster {k} (membership {mu[k]:.0%})."),
        dict(stage="Stage 2 - XAI", text=f"SHAP ranked which symptoms pushed the patient into cluster {k}; the IMM tree gave a {len(path)}-step rule (agrees with the network on {ex.fidelity:.0%} of training patients); counterfactuals show what would change the cluster."),
        dict(stage="Stage 3 - Symbolic reasoning", text=f"The cluster centroid was matched against the disease-symptom knowledge base; the 8 closest diseases were re-scored with this patient's symptoms (60% similarity, 40% rule agreement). Status: {g['status']}."),
    ]
    return dict(cluster=k, membership=float(mu[k]), l1_node=n1, status=g["status"], ranked=g["ranked"],
                shap=[dict(symptom=nice(s), value=v) for s, v in shap_l],
                signature=[dict(symptom=nice(s), value=v) for s, v in ex.signature(k)],
                rule=[f"{nice(ex.vocab[j])} PRESENT" for j, p in path if p] + ["Not present: " + ", ".join(nice(ex.vocab[j]) for j, p in path if not p)],
                tree_leaf=leaf, counterfactuals=cfs, red_flags=[nice(s) for s in srm.red_flags(r.symptoms)],
                stages=stages, disclaimer="Possible conditions only - not a diagnosis. See a clinician.")

def need_key(x_api_key: str = Header(None)):
    if not os.environ.get("ADMIN_KEY") or x_api_key != os.environ["ADMIN_KEY"]: raise HTTPException(status_code=401)

@app.post("/api/ingest", dependencies=[Depends(need_key)])
def ingest(r: Req):
    """Growth demo: feed one new patient to Layer 1 and report what the network did."""
    x, m = S["enc"].encode(r.symptoms, r.absent); n = S["net"].l1
    before = len(n.w); ev = n.learn(x, m)
    return dict(event=ev, l1_nodes_before=before, l1_nodes_after=len(n.w))

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def home(): return FileResponse("static/index.html")