# PSN — Guide d’entraînement / Construction (handoff Agnes)

**Destinataire :** Agnes  
**Projet :** Phase Siren Network (PSN)  
**Repo :** https://github.com/AFKmoney/phase-siren-network  
**Dataset cible :** https://huggingface.co/datasets/thefinalboss/fractus-datasets  

Ce document explique **exactement** comment construire (pas « entraîner » au sens gradient) un PSN sur le Fractus Dataset et quoi livrer ensuite.

---

## 1. Ce que PSN est (et n’est pas)

PSN **ne fait pas de backprop**.

Pipeline réel :

1. Chaque token reçoit un **expert primaire** stable (hash du token seul).
2. Une ou plusieurs passes sur le corpus **injectent** directement les co-occurrences dans les matrices de poids de cet expert.
3. Génération = n-gram local + signal de l’expert spécialisé (freq + projection SVD) + légère modulation Kuramoto.

Il n’y a **pas** de loss, pas d’optimizer, pas de learning rate classique.  
« Entraîner » = **construire** les poids en une passe (ou 2-3).

Code actif :
- `psn/model.py` → classes `HashDistributor` + `PSN`
- `psn/data.py` → chargeur de corpus
- `scripts/talk.py` → génération interactive
- `scripts/eval.py` → métriques

---

## 2. Dataset Fractus (HF)

Repo : `thefinalboss/fractus-datasets`  
Taille totale ≈ **28 Go / 3–4 B tokens**.

Contenu principal :
- `neuro_paradigms_1b/` — paradigmes neuroscience → architecture
- `cognitive_skills/` — reasoning, coding, etc. (jsonl)
- `neuro_code_math/`
- `gutenberg/esoteric.jsonl`
- `dictionaries/wordnet.jsonl`
- `repos/` — code des repos de thefinalboss
- plusieurs `.pt` pré-tokenisés (GPT-2 BPE)

**Ne télécharge pas tout d’un coup** sauf si tu as 50 Go+ libres et beaucoup de temps.

### Stratégie de subset recommandée (ordre de priorité)

1. `cognitive_skills/**/*.jsonl` (reasoning + coding)
2. `gutenberg/esoteric.jsonl`
3. `dictionaries/wordnet.jsonl`
4. Un échantillon de `neuro_paradigms_1b/*.jsonl.gz` (10–30 fichiers)
5. `repos/**/*.jsonl` si pertinent

Objectif réaliste pour un premier modèle : **200 M – 1 B caractères** de texte propre.

---

## 3. Pipeline exact à exécuter

### 3.1 Prérequis

```bash
pip install numpy huggingface_hub
# token HF avec read (et write si tu push le modèle)
export HF_TOKEN=hf_...
```

### 3.2 Télécharger un subset et le transformer en texte

Créer un script `scripts/build_fractus_corpus.py` (ou exécuter en interactif) qui :

1. Liste les fichiers via `list_repo_files("thefinalboss/fractus-datasets", repo_type="dataset")`
2. Télécharge les fichiers choisis avec `hf_hub_download`
3. Parse les schémas JSONL (clés fréquentes : `instruction`/`response`, `messages`, `content`, `s`, `text`, `definition`)
4. Concatène tout en un seul fichier texte UTF-8 : `data/fractus_corpus.txt`

Attention LFS : sans token valide les fichiers peuvent arriver vides (0 byte).

### 3.3 Construire le PSN

```python
from psn import PSN, load_corpus
import numpy as np

# soit via load_corpus si tu as mis le fichier dans data/
# soit manuellement :

text = open("data/fractus_corpus.txt", encoding="utf-8").read()
chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
token_ids = np.array([c2i[c] for c in text], dtype=np.int32)
V = len(chars)

model = PSN(
    num_experts=64,      # ou 96 / 128 si plus de data
    weight_dim=128,      # ou 192 / 256
    vocab_size=V,
    kuramoto_K=2.4,
    seed=42,
)
model.build(token_ids, V, passes=2, primary_mass=0.92)
```

Paramètres importants :
- `primary_mass=0.90–0.95` → spécialisation forte
- `passes=2` (suffisant pour la plupart des cas)
- Plus d’experts + dim plus large si le corpus dépasse ~1 B caractères

### 3.4 Sauvegarder les poids

```python
import numpy as np, json

np.savez_compressed(
    "results/psn_fractus_weights.npz",
    expert_weights=model.distributor.expert_weights,
    expert_biases=model.distributor.expert_biases,
    token_freq=model.distributor.token_freq,
    embedding=model.embedding,
    expert_heads=model.expert_heads,
    expert_proj=model.expert_proj,
    expert_phases=model.expert_phases,
    expert_omegas=model.expert_omegas,
)

meta = {
    "num_experts": model.num_experts,
    "weight_dim": model.weight_dim,
    "vocab_size": V,
    "primary_mass": 0.92,
    "passes": 2,
    "char_to_idx": c2i,
    "idx_to_char": {str(k): v for k, v in i2c.items()},
    "dataset": "thefinalboss/fractus-datasets (subset)",
}
json.dump(meta, open("results/psn_fractus_meta.json", "w"), ensure_ascii=False, indent=2)
```

### 3.5 Push sur Hugging Face

Créer un repo modèle, ex. `thefinalboss/psn-fractus` :

```bash
huggingface-cli repo create psn-fractus --type model
# ou via l’UI

# puis
huggingface-cli upload thefinalboss/psn-fractus results/psn_fractus_weights.npz
huggingface-cli upload thefinalboss/psn-fractus results/psn_fractus_meta.json
# + model card + code de chargement
```

Inclure dans le model card :
- que c’est une **construction par hash**, pas un entraînement gradient
- taille du subset utilisé
- hyperparamètres (experts, dim, primary_mass)
- comment recharger et générer

---

## 4. Chargement + génération (pour validation)

```python
# recharger
data = np.load("results/psn_fractus_weights.npz")
# reconstruire un objet PSN et assigner les tableaux
# puis :
encode = lambda s: [c2i.get(c, 0) for c in s]
decode = lambda ids: "".join(i2c.get(i, "") for i in ids)
print(model.generate("Explain hippocampal replay", encode, decode, length=300, temperature=0.45, expert_strength=0.80))
```

---

## 5. Métriques à regarder

Lancer `python scripts/eval.py` (adapter le corpus) et vérifier :

- `the_rate` (doit baisser si le corpus est plus riche)
- domain / neuro / coding word recall
- diversité n-gram
- spécialisation (tokens uniques par expert ≈ 5–20, pas 60+)

---

## 6. Ce qu’Agnes doit livrer

1. `data/fractus_corpus.txt` (ou indication précise du subset)
2. `results/psn_fractus_weights.npz` + `psn_fractus_meta.json`
3. Repo HF `thefinalboss/psn-fractus` (ou nom choisi) avec :
   - poids
   - meta
   - model card
   - petit script `load_and_generate.py`
4. Note courte des hyperparamètres et de la taille réelle du corpus utilisé

---

## 7. Pièges connus

- Fichiers HF à **0 byte** → token manquant ou LFS non résolu.
- Vocab trop petit / trop de caractères de contrôle → nettoyer le texte.
- `the the the` dominant → augmenter `expert_strength` (0.75–0.85) + repetition penalty (déjà dans le code).
- Mémoire : `expert_weights` est `(E, D, D)`. 128 experts × 256 dim ≈ raisonnable ; au-delà surveiller la RAM.

---

## 8. État actuel du repo (août 2026)

- Structure clean (`data/`, `psn/model.py`, `scripts/talk.py`, `scripts/eval.py`)
- PSN avec **spécialisation primaire** déjà implémentée
- Testé sur Shakespeare + Bible + Paradise Lost + Meditations (~6.3 M tokens)
- Prêt à recevoir le corpus Fractus

Dès que le corpus Fractus est en local sous forme texte, la construction prend de quelques secondes à quelques minutes selon la taille.

---

**Fin de la note.**  
Agnes : construis sur Fractus, sauvegarde les poids, push sur HF, et documente le subset exact utilisé.
