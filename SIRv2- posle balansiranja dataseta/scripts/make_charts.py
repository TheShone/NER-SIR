#!/usr/bin/env python3
"""Grafikoni rezultata (ćirilica) za BALANSIRANI skup — runda 2. Izlaz u figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=150); plt.close(fig)
plt.rcParams.update({"font.size": 11, "axes.axisbelow": True})

MODELI = ["Qwen3-4B\nинстр.", "Qwen3-4B\nмислећи", "Qwen3-30B\nинстр.", "Qwen3-30B\nмислећи", "NER4Legal"]
x = np.arange(len(MODELI)); w = 0.38
BOJE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

def anno(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0,2), ha="center", fontsize=8)

# Grafikon 1 — F1 strogo/ublaženo
strogo=[0.423,0.491,0.540,0.335,0.445]; relaxed=[0.535,0.567,0.633,0.362,0.749]
fig,ax=plt.subplots(figsize=(8.4,4.8))
b1=ax.bar(x-w/2,strogo,w,label="F1 (строго)",color="#4C72B0")
b2=ax.bar(x+w/2,relaxed,w,label="F1 (ублажено)",color="#DD8452")
ax.set_ylabel("F1 мера"); ax.set_ylim(0,0.85); ax.grid(axis="y",alpha=0.3)
ax.set_title("Укупна F1 мера по систему (балансиран скуп, подскуп 450)")
ax.set_xticks(x); ax.set_xticklabels(MODELI); ax.legend(); anno(ax,list(b1)+list(b2))
save(fig,"grafikon1_f1.png")

# Grafikon 2 — P/R strogo
P=[0.538,0.781,0.644,0.866,0.536]; R=[0.348,0.358,0.465,0.207,0.380]
fig,ax=plt.subplots(figsize=(8.4,4.8))
b1=ax.bar(x-w/2,P,w,label="Прецизност",color="#55A868")
b2=ax.bar(x+w/2,R,w,label="Одзив",color="#C44E52")
ax.set_ylabel("вредност"); ax.set_ylim(0,1.0); ax.grid(axis="y",alpha=0.3)
ax.set_title("Прецизност и одзив по систему (строга метрика)")
ax.set_xticks(x); ax.set_xticklabels(MODELI); ax.legend(); anno(ax,list(b1)+list(b2),"{:.2f}")
save(fig,"grafikon2_pr.png")

# Grafikon 3 — F1 po tipu (ublaženo)
TIP=["COURT","DATE","DECISION","LAW","MONEY","OFF.GAZ.","PERSON","REFER."]
data={"Qwen3-4B инстр.":[0.620,0.578,0.063,0.455,0.761,0.556,0.328,0.788],
      "Qwen3-4B мислећи":[0.561,0.663,0.244,0.531,0.789,0.632,0.519,0.656],
      "Qwen3-30B инстр.":[0.668,0.957,0.088,0.493,0.848,0.670,0.491,0.686],
      "Qwen3-30B мислећи":[0.333,0.404,0.097,0.217,0.613,0.183,0.486,0.417],
      "NER4Legal":[0.765,0.993,0.000,0.297,0.969,0.960,0.917,0.815]}
xt=np.arange(len(TIP)); n=len(data); bw=0.16
fig,ax=plt.subplots(figsize=(11,5.4))
for i,(name,vals) in enumerate(data.items()):
    ax.bar(xt+(i-(n-1)/2)*bw,vals,bw,label=name,color=BOJE[i])
ax.set_ylabel("F1 мера (ублажено)"); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3)
ax.set_title("F1 мера по типу ентитета (ублажена метрика)")
ax.set_xticks(xt); ax.set_xticklabels(TIP,rotation=15); ax.legend(ncol=3,fontsize=9)
save(fig,"grafikon3_tip.png")

# Grafikon 4 — balansiranje pre/posle
TIPB=["LAW","OFF.GAZ.","COURT","DECISION","DATE","REFER.","MONEY","PERSON"]
pre=[1409,787,77,71,66,64,48,5]; posle=[810,613,417,367,343,262,229,248]
xb=np.arange(len(TIPB));
fig,ax=plt.subplots(figsize=(9.5,4.8))
b1=ax.bar(xb-w/2,pre,w,label="пре балансирања (282:1)",color="#BFBFBF")
b2=ax.bar(xb+w/2,posle,w,label="после балансирања (≈3,5:1)",color="#4C72B0")
ax.set_ylabel("број ентитета"); ax.grid(axis="y",alpha=0.3)
ax.set_title("Расподела ентитета пре и после балансирања скупа")
ax.set_xticks(xb); ax.set_xticklabels(TIPB,rotation=15); ax.legend()
save(fig,"grafikon4_balans.png")

# Grafikon 5 — preciznost-odziv scatter (strogo)
fig,ax=plt.subplots(figsize=(7.2,5.4))
names=["Qwen3-4B инстр.","Qwen3-4B мислећи","Qwen3-30B инстр.","Qwen3-30B мислећи","NER4Legal"]
for i,nm in enumerate(names):
    ax.scatter(P[i],R[i],s=150,color=BOJE[i],zorder=3)
    ax.annotate(nm,(P[i],R[i]),textcoords="offset points",xytext=(8,4),fontsize=9)
ax.set_xlabel("прецизност"); ax.set_ylabel("одзив"); ax.set_xlim(0,1.0); ax.set_ylim(0,0.6)
ax.grid(alpha=0.3); ax.set_title("Прецизност наспрам одзива (строга метрика)")
save(fig,"grafikon5_pr_scatter.png")

print("Napravljeni grafikoni u", OUT, ":", os.listdir(OUT))
