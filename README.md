# AnalyseSang
Pipeline d’extraction et d’analyse de bilans sanguins (PDF → CSV → Excel)

## 📌 Objectif
Automatiser le traitement de bilans sanguins PDF pour générer un Excel structuré, lisible et orienté suivi médical.

- Extraction des valeurs à partir des PDF (y compris les **antériorités** listées).  
- Conversion et **unification des unités** (ex. cholestérol en mmol/L).  
- Classification des marqueurs dans des **groupes médicaux** (lipides, glycémie, rénal, thyroïde…).  
- Génération d’un **Excel final** avec :
  - `Hors_norme` : marqueurs hors norme, surlignés en jaune, colonnes auto-ajustées.  
  - `Dans_la_norme` : marqueurs toujours normaux.  
  - `Courbe_{marqueur}` : graphiques d’évolution (points reliés, lignes-cibles LDL/HDL/etc., lien de retour).  

---

