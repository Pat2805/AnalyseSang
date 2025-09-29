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
## A Faire

Agent extrateur : 
- Gestion des antécédants : 
si une valeur existe déjà et qu'on trouve un antécédant, il n'est pas ajouté puisque déjà enregistré. De même, si un antécédant est déjà enregistré et qu'une valeur originale est lue dans un fichier, il ne faut pas créer deux entrées mais modifier l'actuelle par celle lue dans le fichier et ne pas la tagger comme antécédant. Pour résumer, les valeurs antécédantes seront ignorées si le bilan sanguin original est présent dans les documents. Dans le cas où les valeurs antécédantes et non antécédantes sont différentes, il faut garder les valeurs originales et enregistrer les détails du problème sur un fichier log d'erreur antécédants spécifique.
- Gestion des unités avec les antécédants : vérifier que l'unification des unités est bien appliquée lorsque les valeurs viennent des antécédants.
- Gestion des valeurs multiples (celles avec % d'abord puis valeurs elles même notamment) : de base, si on a déjà des valeurs de ce marqueur, utiliser les mêmes unités. Sinon regarder s'il y a des références min max sur le doc, si oui, utiliser les valeurs/unités correspondantes. Sinon, utiliser les valeurs/unités les plus répandues.
- Afficher les rejets dans les logs pour comprendre les problemes.
- Bug pour la provenance qui dit tj antécédants : faire une analyse plus fine. 

  
Creation d'un agent correcteur  : corrections sur les données obtenues.
- Harmonisation des marqueurs_key : Verifier qu'il n'y a pas de doublon possibles. Si détection, verification sur les pdf concernés et par prompt IA pour harmoniser.
  Faire les vérifications puis proposer à la fin à l'utilisateur les modifications proposées. Demander aussi à l'utilisateur si il faut retenir son choix pour la prochaine fois (ignorer l'alerte ou appliquer la correction).
  Pour les fausses alertes, demander à l'utilisateur si il faut les ignorer les prochaines fois.
  Faire les modifications demandées et mettre à jour un fichier de config qui gardera les choix utilisateur ci dessus. 
  Faire un fichier log specifique sur les alertes levées et les choix utilisateurs, et les modifications apportées.
  
- Harmonisation des unités source / cible. :

- Gestion des valeurs min / max : 

  
