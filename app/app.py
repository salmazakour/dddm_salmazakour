"""
DDDM E-Commerce — Flask Backend
================================
Lance depuis la racine du projet :  python run.py
Dashboard : http://localhost:5000

Structure CSV attendue :
  transactions.csv : InvoiceNo, CustomerID, ProductID, ProductName, Category,
                     Quantity, UnitPrice, Discount, Revenue, InvoiceDate, Country, ReturnFlag
  reviews.csv      : ReviewID, ProductID, ProductName, Category, Rating,
                     ReviewText, ReviewDate, HelpfulVotes, VerifiedPurchase
"""

import pickle, warnings
from pathlib import Path
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PKL_PATH = DATA_DIR / "pipeline_output.pkl"

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(tx_path, rv_path=None):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline as SKPipeline

    # ── 1. TRANSACTIONS ────────────────────────────────────────
    # Auto-détection encodage : supporte UTF-8 et ISO-8859-1 (UCI original)
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(tx_path, encoding=enc, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        df = pd.read_csv(tx_path, encoding="utf-8", errors="replace", low_memory=False)

    # Normalise colonnes (supporte UCI original ET notre format)
    df.rename(columns={
        "StockCode": "ProductID",
        "Description": "ProductName",
    }, inplace=True)

    df["Quantity"]  = pd.to_numeric(df.get("Quantity",  pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df.dropna(subset=["CustomerID", "ProductName"], inplace=True)
    df.drop_duplicates(inplace=True)

    # Revenue : utiliser colonne existante ou recalculer
    if "Revenue" not in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    else:
        df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(
            df["Quantity"] * df["UnitPrice"]
        )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
    df["CustomerID"]  = df["CustomerID"].astype(str)
    df["ProductID"]   = df["ProductID"].astype(str)

    # ── 2. REVIEWS ─────────────────────────────────────────────
    rev_map = {}
    has_reviews = False
    if rv_path and Path(rv_path).exists():
        for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin-1"):
            try:
                rv = pd.read_csv(rv_path, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            rv = pd.read_csv(rv_path, encoding="utf-8", errors="replace", low_memory=False)
        # supporte Amazon Fine Food ET notre format
        rv.rename(columns={
            "ProductId": "ProductID", "Id": "ReviewID",
            "Score": "Rating", "Text": "ReviewText",
            "HelpfulnessNumerator": "HelpfulVotes",
        }, inplace=True)
        rv["Rating"]    = pd.to_numeric(rv.get("Rating", pd.Series(dtype=float)), errors="coerce")
        rv["ProductID"] = rv["ProductID"].astype(str)
        rv = rv[rv["Rating"].between(1, 5)].drop_duplicates()
        for pid, grp in rv.groupby("ProductID")["Rating"]:
            rats = grp.values
            rev_map[str(pid)] = {
                "avg":    float(np.mean(rats)),
                "n":      int(len(rats)),
                "pctPos": float((rats >= 4).mean() * 100),
                "pctNeg": float((rats <= 2).mean() * 100),
            }
        has_reviews = bool(rev_map)

    # ── 3. PRODUITS ────────────────────────────────────────────
    cat_col_exists = "Category" in df.columns

    def guess_cat(name):
        n = str(name).lower()
        if any(k in n for k in ["bag","shirt","dress","trouser","coat","jacket","shoe","sock","scarf","hoodie","jean","pant"]): return "Clothing"
        if any(k in n for k in ["book","guide","manual","diary","novel"]): return "Books"
        if any(k in n for k in ["cream","serum","soap","perfume","beauty","lipstick","mascara","moisturizer"]): return "Beauty"
        if any(k in n for k in ["toy","game","doll","puzzle","lego","teddy","rc car","board game"]): return "Toys"
        if any(k in n for k in ["cable","charger","battery","phone","electronic","led","webcam","keyboard","mouse","speaker","headphone"]): return "Electronics"
        if any(k in n for k in ["sport","gym","yoga","fitness","ball","dumbbell","resistance","protein","tracker"]): return "Sports"
        return "Home & Kitchen"

    rev_max = df.groupby("ProductID")["Revenue"].sum().max() or 1
    rng = np.random.default_rng(42)
    products = []

    for pid, g in df.groupby("ProductID"):
        spid   = str(pid)
        rev    = float(g["Revenue"].sum())
        qty    = int(g["Quantity"].sum())
        ords   = int(g["InvoiceNo"].nunique())
        custs  = int(g["CustomerID"].nunique())
        avg_p  = float(g["UnitPrice"].mean())
        name   = str(g["ProductName"].iloc[0])
        cat    = str(g["Category"].iloc[0]) if cat_col_exists else guess_cat(name)
        top_co = str(g["Country"].value_counts().index[0]) if "Country" in g.columns and len(g) > 0 else ""

        if spid in rev_map:
            rd = rev_map[spid]
            avg_rat, nb_rev, pct_pos, pct_neg = rd["avg"], rd["n"], rd["pctPos"], rd["pctNeg"]
        else:
            pct_rank = rev / rev_max
            avg_rat  = float(np.clip(2.5 + pct_rank * 1.8 + rng.normal(0, 0.35), 1, 5))
            nb_rev   = int(rng.integers(5, 80))
            pct_pos  = float(np.clip((avg_rat - 1) / 4 * 70 + rng.uniform(0, 20), 10, 95))
            pct_neg  = float(np.clip(100 - pct_pos - rng.uniform(5, 15), 0, 55))

        products.append(dict(
            pid=spid, name=name, category=cat,
            revenue=rev, qty=qty, orders=ords, customers=custs,
            avgPrice=round(avg_p, 2), topCountry=top_co,
            avgRating=round(avg_rat, 2), nbReviews=nb_rev,
            pctPos=round(pct_pos, 1), pctNeg=round(pct_neg, 1),
        ))

    # ── 4. SCORE COMPOSITE ─────────────────────────────────────
    def minmax(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]

    revs = minmax([p["revenue"]   for p in products])
    rats = minmax([p["avgRating"] for p in products])
    vols = minmax([p["orders"]    for p in products])

    for i, p in enumerate(products):
        p["scoreRevenue"] = round(revs[i] * 40, 2)
        p["scoreSat"]     = round(rats[i] * 35, 2)
        p["scoreVol"]     = round(vols[i] * 25, 2)
        p["score"]        = round(p["scoreRevenue"] + p["scoreSat"] + p["scoreVol"], 2)

    # ── BCG et décision par rang percentile ────────────────────
    # Garantit une distribution équilibrée quel que soit le dataset
    n_prod = len(products)
    # Rang revenue : position du produit dans le classement revenue
    sorted_rev = sorted(range(n_prod), key=lambda i: products[i]["revenue"])
    sorted_rat = sorted(range(n_prod), key=lambda i: products[i]["avgRating"])
    sorted_scr = sorted(range(n_prod), key=lambda i: products[i]["score"])

    rev_pct = [0.0] * n_prod
    rat_pct = [0.0] * n_prod
    scr_pct = [0.0] * n_prod
    for rank, idx in enumerate(sorted_rev):
        rev_pct[idx] = rank / max(n_prod - 1, 1) * 100
    for rank, idx in enumerate(sorted_rat):
        rat_pct[idx] = rank / max(n_prod - 1, 1) * 100
    for rank, idx in enumerate(sorted_scr):
        scr_pct[idx] = rank / max(n_prod - 1, 1) * 100

    for i, p in enumerate(products):
        # Décision : top 20% = promouvoir, bottom 20% = retirer
        p["decision"] = ("promote" if scr_pct[i] >= 80 else
                         "retire"  if scr_pct[i] <= 20 else "watch")
        # BCG : médiane revenue × médiane rating → 4 quadrants ~égaux
        hi_rev = rev_pct[i] >= 50
        hi_rat = rat_pct[i] >= 50
        if hi_rev and hi_rat:
            p["bcg"] = "stars"
        elif hi_rev and not hi_rat:
            p["bcg"] = "cash"
        elif not hi_rev and hi_rat:
            p["bcg"] = "qm"
        else:
            p["bcg"] = "dogs"

    # ── 5. CLUSTERING ──────────────────────────────────────────
    feats = np.array([[p["revenue"], p["qty"], p["orders"],
                       p["avgRating"], p["pctPos"]] for p in products], dtype=float)
    feats[:, 0] = np.log1p(feats[:, 0])
    feats[:, 1] = np.log1p(feats[:, 1])
    sc = StandardScaler()
    Xs = sc.fit_transform(feats)

    inertias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
        km.fit(Xs)
        inertias.append({"k": k, "inertia": round(km.inertia_, 1)})

    km4 = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=200)
    labels4 = km4.fit_predict(Xs)
    for i, p in enumerate(products):
        p["cluster"] = int(labels4[i])

    # ── 6. MODÈLES PRÉDICTIFS ──────────────────────────────────
    FEAT_COLS = ["revenue", "qty", "orders", "customers",
                 "avgPrice", "avgRating", "pctPos", "pctNeg"]
    X = np.array([[p[f] for f in FEAT_COLS] for p in products], dtype=float)
    X[:, 0] = np.log1p(X[:, 0])
    X[:, 1] = np.log1p(X[:, 1])
    y = np.array([1 if p["decision"] == "promote" else 0 for p in products])

    model_results = []
    feat_imp = []
    if len(set(y)) > 1 and len(y) >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for mname, clf in [
            ("Régression Logistique", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest",         RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Gradient Boosting",     GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]:
            pipe = SKPipeline([("sc", StandardScaler()), ("m", clf)])
            cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
            pipe.fit(X_tr, y_tr)
            yp    = pipe.predict(X_te)
            yprob = pipe.predict_proba(X_te)[:, 1]
            model_results.append({
                "name":      mname,
                "cv_auc":    round(float(cv_auc.mean()), 3),
                "cv_std":    round(float(cv_auc.std()),  3),
                "test_auc":  round(float(roc_auc_score(y_te, yprob)), 3),
                "f1":        round(float(f1_score(y_te, yp)), 3),
                "precision": round(float(precision_score(y_te, yp, zero_division=0)), 3),
                "recall":    round(float(recall_score(y_te, yp,    zero_division=0)), 3),
            })
            if mname == "Random Forest":
                fi = pipe.named_steps["m"].feature_importances_
                feat_imp = sorted(
                    [{"feat": FEAT_COLS[i], "importance": round(float(fi[i]), 4)}
                     for i in range(len(FEAT_COLS))],
                    key=lambda x: -x["importance"]
                )

    # ── 7. KPIs GLOBAUX ────────────────────────────────────────
    monthly    = df.groupby("YearMonth")["Revenue"].sum().reset_index().sort_values("YearMonth")
    by_country = df.groupby("Country")["Revenue"].sum().nlargest(10) if "Country" in df.columns else {}
    by_cat     = df.groupby("Category")["Revenue"].sum() if cat_col_exists else \
                 df.copy().assign(Category=df["ProductName"].apply(guess_cat)).groupby("Category")["Revenue"].sum()

    p95        = float(df["UnitPrice"].quantile(0.95))
    hist_p, bin_e = np.histogram(df[df["UnitPrice"] <= p95]["UnitPrice"].values, bins=20)

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_rev   = df.groupby("DayOfWeek")["Revenue"].sum().reindex(dow_order).fillna(0)

    if has_reviews:
        all_r = [rd["avg"] for rd in rev_map.values() for _ in range(rd["n"])]
        avg_global = float(np.mean(all_r)) if all_r else 0
        pct_pos_g  = float(np.mean([rd["pctPos"] for rd in rev_map.values()]))
        pct_neg_g  = float(np.mean([rd["pctNeg"] for rd in rev_map.values()]))
        tot_rev_n  = sum(rd["n"] for rd in rev_map.values())
    else:
        avg_global = float(np.mean([p["avgRating"] for p in products]))
        pct_pos_g  = float(np.mean([p["pctPos"]    for p in products]))
        pct_neg_g  = float(np.mean([p["pctNeg"]    for p in products]))
        tot_rev_n  = int(sum(p["nbReviews"]         for p in products))

    # ── 8. TESTS STATISTIQUES ──────────────────────────────────
    stat_tests = []
    if "Country" in df.columns:
        uk  = df[df["Country"] == "United Kingdom"]["Revenue"]
        non = df[df["Country"] != "United Kingdom"]["Revenue"]
        if len(uk) > 1 and len(non) > 1:
            t, p = stats.ttest_ind(uk, non)
            stat_tests.append({
                "test": "T-test Revenue UK vs Non-UK",
                "stat": round(float(t), 3),
                "pvalue": round(float(p), 4),
                "significant": bool(p < 0.05)
            })
    corr, pc = stats.pearsonr(
        df["Revenue"].clip(upper=df["Revenue"].quantile(0.99)),
        df["Quantity"].clip(upper=df["Quantity"].quantile(0.99))
    )
    stat_tests.append({
        "test": "Corrélation Pearson Revenue / Quantité",
        "stat": round(float(corr), 3),
        "pvalue": round(float(pc), 4),
        "significant": bool(pc < 0.05)
    })

    # ── 9. RÉSULTAT ────────────────────────────────────────────
    return {
        "meta": {
            "n_rows":      int(len(df)),
            "n_products":  int(len(products)),
            "n_customers": int(df["CustomerID"].nunique()),
            "n_orders":    int(df["InvoiceNo"].nunique()),
            "date_min":    str(df["InvoiceDate"].min().date()),
            "date_max":    str(df["InvoiceDate"].max().date()),
            "has_reviews": has_reviews,
        },
        "kpis": {
            "total_revenue":   round(float(df["Revenue"].sum()), 2),
            "total_orders":    int(df["InvoiceNo"].nunique()),
            "total_customers": int(df["CustomerID"].nunique()),
            "avg_basket":      round(float(df.groupby("InvoiceNo")["Revenue"].sum().mean()), 2),
            "avg_unit_price":  round(float(df["UnitPrice"].mean()), 2),
            "avg_rating":      round(avg_global, 2),
            "pct_positive":    round(pct_pos_g, 1),
            "pct_negative":    round(pct_neg_g, 1),
            "total_reviews":   tot_rev_n,
        },
        "monthly":    [{"month": r["YearMonth"], "revenue": round(float(r["Revenue"]), 2)}
                       for _, r in monthly.iterrows()],
        "by_country": [{"country": k, "revenue": round(float(v), 2)}
                       for k, v in by_country.items()],
        "by_category":[{"category": k, "revenue": round(float(v), 2)}
                       for k, v in by_cat.items()],
        "dow_revenue":[{"day": d, "revenue": round(float(dow_rev[d]), 2)}
                       for d in dow_order if d in dow_rev.index],
        "price_hist": {"counts": [int(x) for x in hist_p],
                       "edges":  [round(float(x), 2) for x in bin_e]},
        "products":   products,
        "elbow":      inertias,
        "models":     model_results,
        "feat_imp":   feat_imp,
        "stat_tests": stat_tests,
        "bcg_counts": {
            "stars": sum(1 for p in products if p["bcg"] == "stars"),
            "cash":  sum(1 for p in products if p["bcg"] == "cash"),
            "qm":    sum(1 for p in products if p["bcg"] == "qm"),
            "dogs":  sum(1 for p in products if p["bcg"] == "dogs"),
        },
        "decision_counts": {
            "promote": sum(1 for p in products if p["decision"] == "promote"),
            "watch":   sum(1 for p in products if p["decision"] == "watch"),
            "retire":  sum(1 for p in products if p["decision"] == "retire"),
        },
    }


# ─────────────────────────────────────────────────────────────
#  CACHE & CHARGEMENT
# ─────────────────────────────────────────────────────────────
_cache = {"data": None, "error": None}

def load_or_compute():
    tx = DATA_DIR / "transactions.csv"
    rv = DATA_DIR / "reviews.csv"
    if not tx.exists():
        return None, "transactions.csv introuvable dans data/"
    try:
        data = run_pipeline(str(tx), str(rv) if rv.exists() else None)
        with open(PKL_PATH, "wb") as f:
            pickle.dump(data, f)
        return data, None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()

def get_data():
    if _cache["data"] is None:
        if PKL_PATH.exists():
            with open(PKL_PATH, "rb") as f:
                _cache["data"] = pickle.load(f)
        else:
            _cache["data"], _cache["error"] = load_or_compute()
    return _cache["data"], _cache["error"]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    data, err = get_data()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({"ok": True, "meta": data["meta"]})

@app.route("/api/kpis")
def api_kpis():
    data, _ = get_data()
    return jsonify(data["kpis"] if data else {}), (200 if data else 500)

@app.route("/api/overview")
def api_overview():
    data, _ = get_data()
    if not data: return jsonify({}), 500
    return jsonify({
        "monthly":     data["monthly"],
        "by_country":  data["by_country"],
        "by_category": data["by_category"],
        "dow_revenue": data["dow_revenue"],
        "price_hist":  data["price_hist"],
        "stat_tests":  data["stat_tests"],
    })

@app.route("/api/products")
def api_products():
    data, _ = get_data()
    if not data: return jsonify([]), 500
    cat   = request.args.get("category", "")
    sort  = request.args.get("sort", "revenue")
    limit = int(request.args.get("limit", 50))
    prods = [p for p in data["products"] if not cat or p["category"] == cat]
    prods = sorted(prods, key=lambda p: p.get(sort, 0), reverse=True)
    return jsonify(prods[:limit])

@app.route("/api/products/all")
def api_products_all():
    data, _ = get_data()
    return jsonify(data["products"] if data else [])

@app.route("/api/sentiment")
def api_sentiment():
    data, _ = get_data()
    if not data: return jsonify({}), 500
    prods = data["products"]
    by_cat = {}
    for p in prods:
        by_cat.setdefault(p["category"], []).append(p["avgRating"])
    cat_avg = sorted(
        [{"category": k, "avgRating": round(sum(v)/len(v), 2)} for k, v in by_cat.items()],
        key=lambda x: -x["avgRating"]
    )
    top10  = sorted([p for p in prods if p["nbReviews"] >= 5], key=lambda x: -x["avgRating"])[:10]
    flop10 = sorted([p for p in prods if p["nbReviews"] >= 5], key=lambda x:  x["avgRating"])[:10]
    return jsonify({
        "kpis":   {k: data["kpis"][k] for k in
                   ("avg_rating","pct_positive","pct_negative","total_reviews")},
        "by_cat": cat_avg,
        "top10":  top10,
        "flop10": flop10,
    })

@app.route("/api/decisions")
def api_decisions():
    data, _ = get_data()
    if not data: return jsonify({}), 500
    prods   = data["products"]
    promote = sorted([p for p in prods if p["decision"]=="promote"], key=lambda x: -x["score"])[:10]
    watch   = sorted([p for p in prods if p["decision"]=="watch"],   key=lambda x: -x["score"])[:10]
    retire  = sorted([p for p in prods if p["decision"]=="retire"],  key=lambda x:  x["score"])[:10]
    return jsonify({
        "bcg_counts":      data["bcg_counts"],
        "decision_counts": data["decision_counts"],
        "promote": promote,
        "watch":   watch,
        "retire":  retire,
        "scores":  [p["score"] for p in prods],
        "all_scores_by_bcg": [{"score": p["score"], "bcg": p["bcg"]} for p in prods],  
        "elbow":   data["elbow"],
    })

@app.route("/api/models")
def api_models():
    data, _ = get_data()
    if not data: return jsonify({}), 500
    return jsonify({"models": data["models"], "feat_imp": data["feat_imp"]})

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    _cache["data"] = _cache["error"] = None
    if PKL_PATH.exists(): PKL_PATH.unlink()
    data, err = load_or_compute()
    _cache["data"], _cache["error"] = data, err
    if err: return jsonify({"ok": False, "error": err}), 500
    return jsonify({"ok": True, "meta": data["meta"]})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    saved = {}
    for key in ("transactions", "reviews"):
        f = request.files.get(key)
        if f:
            dest = DATA_DIR / f"{key}.csv"
            f.save(dest)
            saved[key] = dest.name
    if not saved:
        return jsonify({"ok": False, "error": "Aucun fichier reçu"}), 400
    _cache["data"] = _cache["error"] = None
    if PKL_PATH.exists(): PKL_PATH.unlink()
    data, err = load_or_compute()
    _cache["data"], _cache["error"] = data, err
    if err: return jsonify({"ok": False, "error": err, "saved": saved}), 500
    return jsonify({"ok": True, "saved": saved, "meta": data["meta"]})


if __name__ == "__main__":
    print("="*55)
    print("  DDDM — Flask  →  http://localhost:5000")
    print("="*55)
    app.run(debug=True, port=5000, use_reloader=False)
