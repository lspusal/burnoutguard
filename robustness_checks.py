#!/usr/bin/env python3
"""Robustness checks reported in Sections 4.1 and 4.5.

(a) Student-level grouping: OULAD students may hold several registrations, so a
    random split can place the same id_student in train and test. Compares random
    StratifiedKFold against GroupKFold on id_student, full-course and at week 4.
(b) Seed stability of the fairness gaps: repeats the full-course audit over five
    80/20 splits, to show which parity gaps are stable and which are not.
"""
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb, horizons
from pipeline import build_features
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore")

def fit(Xtr,ytr,Xte,seed=42):
    spw=(ytr==0).sum()/(ytr==1).sum()
    Xb,yb=SMOTE(random_state=seed).fit_resample(Xtr,ytr)
    m=xgb.XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.05,subsample=0.8,
      colsample_bytree=0.7,scale_pos_weight=spw,random_state=seed,eval_metric="logloss",
      n_jobs=-1,verbosity=0)
    m.fit(Xb,yb); return m.predict_proba(Xte)[:,1]

# (a) horizonte semana 4: aleatorio vs agrupado por estudiante
vle=pd.read_csv("data/studentVle.csv"); info=pd.read_csv("data/studentInfo.csv")
f,feat=horizons.extract_full(vle,info,4)
X4,y4=f[feat].values,f["y"].values; g4=f["id_student"].values
print("SEMANA 4 — filas %d, estudiantes %d"%(len(y4),len(np.unique(g4))))
for nm,sp in [("aleatorio",list(StratifiedKFold(5,shuffle=True,random_state=42).split(X4,y4))),
              ("agrupado por id_student",list(GroupKFold(5).split(X4,y4,groups=g4)))]:
    a=[roc_auc_score(y4[te],fit(X4[tr],y4[tr],X4[te])) for tr,te in sp]
    print("  %-26s AUROC %.4f +/- %.4f"%(nm,np.mean(a),np.std(a,ddof=1)))

# (b) estabilidad de los gaps de fairness ante el seed (full-course)
data=load_oulad_data("data/"); df,X,y,helpers=build_features(data)
Xv=X.values
def gap(pred,a,yt,kind):
    v=[]
    for lv in np.unique(a):
        m=a==lv
        if m.sum()<20: continue
        if kind=="dp": v.append(pred[m].mean()); continue
        tn,fp,fn,tp=confusion_matrix(yt[m],pred[m],labels=[0,1]).ravel()
        v.append(fp/(fp+tn) if kind=="fpr" and fp+tn else (fn/(fn+tp) if kind=="fnr" and fn+tp else np.nan))
    v=[x for x in v if not np.isnan(x)]
    return max(v)-min(v) if len(v)>1 else np.nan
res={}
for seed in [42,7,123,2024,99]:
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=seed)
    pb=fit(Xv[itr],y[itr],Xv[ite],seed); pred=(pb>=0.5).astype(int); yt=y[ite]
    H=df.iloc[ite][helpers].reset_index(drop=True); Xte=X.iloc[ite]
    gd=(Xte["gender_M"].values==1).astype(int)
    imd=(H["_imd_num"].values<=H["_imd_num"].median()).astype(int)
    age=np.where(H["_age55"].values==1,2,np.where(H["_age3555"].values==1,1,0))
    for an,av in [("gender",gd),("age",age),("imd",imd)]:
        for k in ["dp","fpr","fnr"]:
            res.setdefault((an,k),[]).append(gap(pred,av,yt,k))
    res.setdefault(("_auroc","-"),[]).append(roc_auc_score(yt,pb))
print("\nESTABILIDAD ANTE EL SEED (5 particiones 80/20)")
print("  %-14s %8s %8s %8s"%("","media","mín","máx"))
for k,v in res.items():
    print("  %-14s %8.3f %8.3f %8.3f"%(f"{k[0]} {k[1]}",np.mean(v),min(v),max(v)))
