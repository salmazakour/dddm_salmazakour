# 📊 DDDM — E-Commerce Dashboard

**Module :** Data-Driven Decision Making  
**Sujet :** Analyse des performances produits & satisfaction client e-commerce  
**Stack :** Python · Flask · scikit-learn · Chart.js  

## 🎯 Question Décisionnelle

> **Comment identifier les produits à promouvoir, surveiller ou retirer du catalogue en combinant l'analyse des sentiments clients et les performances commerciales afin d'optimiser le chiffre d'affaires et la satisfaction client ?**

---

## 🗂️ Arborescence du Projet

```
ecommerce_dddm/
│
├── run.py                          ← POINT D'ENTRÉE UNIQUE
├── requirements.txt                ← Dépendances Python
├── README.md                       ← Ce fichier
│
├── data/
│   ├── transactions.csv            ← Dataset 1 : transactions e-commerce
│   ├── reviews.csv                 ← Dataset 2 : avis clients
│   └── pipeline_output.pkl         ← Cache pipeline (auto-généré)
│
├── app/
│   ├── app.py                      ← Backend Flask + pipeline ML
│   └── templates/
│       └── dashboard.html          ← dashboard interactif
│
├── notebooks/
│   └── DDDM_Ecommerce_Analysis.ipynb  ← Notebook analytique 
│
├── docs/                           ← Figures exportées par le notebook
│
└── ab_test/                        ← Documents A/B Test
```

---

## 📦 Datasets Utilisés
                 
| E-Commerce Transactions : [UCI Online Retail](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  `data/transactions.csv` 

 Product Reviews : [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  `data/reviews.csv` 

---

## ⚙️ Installation & Exécution

### 1. Prérequis

- Python 3.9 ou supérieur
- pip

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le Dashboard Flask

```bash
python run.py
```

Ouvrir dans le navigateur : **http://localhost:5000**

### 4. (Optionnel) Lancer le Notebook Jupyter

```bash
pip install jupyter
jupyter notebook notebooks/DDDM_Ecommerce_Analysis.ipynb
```

---

## 🌐 API REST — Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Dashboard interactif |
| GET | `/api/status` | Statut pipeline + métadonnées |
| GET | `/api/kpis`  | KPIs globaux (CA, commandes, clients, panier) |
| GET | `/api/overview` | Revenue mensuel, pays, catégories, tests statistiques |
| GET | `/api/products?category=&sort=revenue&limit=50` | Liste produits filtrée |
| GET | `/api/products/all` | Tous les produits |
| GET | `/api/sentiment` | KPIs satisfaction + top/flop produits |
| GET | `/api/decisions` | Matrice BCG + recommandations |
| GET | `/api/models` | Résultats des 3 modèles ML + feature importance |
| POST| `/api/refresh` | Force recalcul du pipeline |
| POST| `/api/upload` | Upload nouveaux CSV (form-data: `transactions`, `reviews`) |

---

## 📊 Fonctionnalités du Dashboard


|  Vue Globale : KPIs temps réel · Revenue mensuel · Top pays · Catégories · Jour semaine · Tests stat 
|  Produits : Filtres dynamiques · Top 15 · Courbe Pareto 80/20 · Tableau complet trié 
|  Satisfaction : Note moyenne · % positifs/négatifs · Satisfaction par catégorie · Top/Flop 
|  Décisions : Matrice BCG · Score Composite · Listes Promouvoir/Surveiller/Retirer · Recommandations 
|  Modèles : Comparaison 3 modèles · AUC/F1/Precision/Recall · Feature Importance RF 
|  A/B Test : Sliders interactifs · Calcul taille échantillon · Courbe de puissance · Protocole 

---

## 🔄 Pipeline ML

```
transactions.csv + reviews.csv
        ↓
  Nettoyage & Normalisation
        ↓
  Agrégation par Produit
        ↓
  Merge avec Reviews (Rating moyen, % positifs)
        ↓
  Score Composite (40% Revenue + 35% Satisfaction + 25% Volume)
        ↓
  Segmentation BCG (Stars / Cash Cows / Question Marks / Dogs)
        ↓
  Clustering K-Means (k=4)
        ↓
  3 Modèles Prédictifs (LogReg · RandomForest · GradientBoosting)
        ↓
  pipeline_output.pkl  →  API Flask  →  Dashboard
```

