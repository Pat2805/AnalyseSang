# normalize.py
from __future__ import annotations
from typing import Iterable, Dict, Tuple, List
from agent1_extract.models import BilanResult

def normalize_and_dedup(
    items: Iterable[BilanResult],
    units_config: Dict,
    settings: Dict
) -> List[BilanResult]:
    """
    - Unifie les unités selon units_config["units_target"]
    - Convertit si nécessaire via units_config["factors"]
    - Déduplique (date_prelevement, marqueur_key) selon la règle:
      * si occurrence dans l'unité cible existe -> garder celle-là,
      * sinon garder la première et convertir.
    """
    target = units_config.get("units_target", {})
    factors = units_config.get("factors", {})
    dec_sep = settings.get("decimal_separator", ".")  # informatif (on stocke en '.' côté modèle)

    by_key: Dict[Tuple[str, str], List[BilanResult]] = {}
    out: List[BilanResult] = []

    # group by (date, marker)
    for r in items:
        k = (r.date_prelevement, r.marqueur_key)
        by_key.setdefault(k, []).append(r)

    def convert_value(r: BilanResult, to_unit: str) -> BilanResult:
        # Si valeur qualitative -> rien à faire
        if isinstance(r.valeur, str):
            return r.model_copy(update={"unite_cible": None})

        # Si déjà dans la cible -> rien à faire
        if (r.unite_source or "").lower() == (to_unit or "").lower():
            return r.model_copy(update={"unite_cible": to_unit})

        # Cherche facteur dans config (par marqueur)
        fdef = factors.get(r.marqueur_key)
        if not fdef:
            # pas de conversion connue -> on respecte la valeur telle quelle
            return r.model_copy(update={"unite_cible": r.unite_source})

        # Vérifie correspondance from/to
        if fdef.get("from") == r.unite_source and fdef.get("to") == to_unit:
            factor = float(fdef["factor"])
            new_val = round(float(r.valeur) * factor, 2)
            return r.model_copy(update={"valeur": new_val, "unite_cible": to_unit})
        # Sinon non supporté -> on ne convertit pas
        return r.model_copy(update={"unite_cible": r.unite_source})

    for (d, m), arr in by_key.items():
        unit_target = target.get(m)  # peut être None (ex: qualitatif)
        if unit_target:
            # 1) y a-t-il déjà une mesure dans l'unité cible ?
            preferred = next((x for x in arr if (x.unite_source or "").lower() == unit_target.lower()), None)
            if preferred is not None:
                # Garde celle-ci; si valeur numérique, fixe unite_cible = target
                out.append(preferred.model_copy(update={"unite_cible": unit_target}))
            else:
                # 2) pas de mesure dans l'unité cible -> on prend la 1ère et convertit si possible
                base = arr[0]
                out.append(convert_value(base, unit_target))
        else:
            # Pas d'unité cible (qualitatif) -> garder la 1ère telle quelle (unites null ok)
            out.append(arr[0].model_copy(update={"unite_cible": None, "unite_source": arr[0].unite_source}))

    return out
