"""
Point d'entrée unique — lancer depuis la racine :
    python run.py
Dashboard : http://localhost:5000
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
from app import app, get_data

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DDDM E-Commerce — Flask Dashboard")
    print("  http://localhost:5000")
    print("="*55)
    data, err = get_data()
    if err:
        print(f"  ⚠️  Erreur pipeline : {err[:120]}")
    else:
        m = data['meta']
        print(f"  ✅ {m['n_rows']:,} transactions · {m['n_products']:,} produits")
        print(f"  📅 {m['date_min']} → {m['date_max']}")
        print(f"  ⭐ Reviews {'réelles' if m['has_reviews'] else 'synthétiques'}")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
