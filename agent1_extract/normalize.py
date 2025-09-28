from __future__ import annotations
from typing import Iterable, Dict, Tuple, List
from agent1_extract.models import BilanResult

def normalize_and_dedup(
    items: Iterable[BilanResult],
    units_config: Dict,
    settings: Dict
) -> List[BilanResult]:
    """
    Unifie les unités selon units_config["units_target"], convertit si besoin
    via units_config["factors"], et déduplique par (date_prelevement, marqueur_key).
    Règle dédoublonnage:
      - si une occurrence déjà dans l'unité cible existe → garder celle-là;
      - sinon prendre la première et convertir vers l'unité cible si possible.
    Les qualitatives (valeur texte) gardent unités = None.
    """
    target = units_config.get("units_target", {})
    factors = units_config.get("factors", {})

    by_key: Dict[Tuple[str, str], List[BilanResult]] = {}
    out: List[BilanResult] = []

    for r in items:
        k = (r.date_prelevement, r.marqueur_key)
        by_key.setdefault(k, []).append(r)

    def convert_value(r: BilanResult, to_unit: str) -> BilanResult:
        # qualitatif → pas d’unité
        if isinstance(r.valeur, str):
            return r.model_copy(update={"unite_cible": None})
        # déjà dans la cible
        if (r.unite_source or "").lower() == (to_unit or "").lower():
            return r.model_copy(update={"unite_cible": to_unit})
        # facteur de conversion
        fdef = factors.get(r.marqueur_key)
        if fdef and fdef.get("from") == r.unite_source and fdef.get("to") == to_unit:
            factor = float(fdef["factor"])
            new_val = round(float(r.valeur) * factor, 2)
            return r.model_copy(update={"valeur": new_val, "unite_cible": to_unit})
        # pas de conversion connue → conserve tel quel
        return r.model_copy(update={"unite_cible": r.unite_source})

    for (d, m), arr in by_key.items():
        unit_target = target.get(m)  # peut être None
        if unit_target:
            preferred = next((x for x in arr if (x.unite_source or "").lower() == unit_target.lower()), None)
            if preferred is not None:
                out.append(preferred.model_copy(update={"unite_cible": unit_target}))
            else:
                base = arr[0]
                out.append(convert_value(base, unit_target))
        else:
            # qualitatif ou pas d'unité cible définie
            out.append(arr[0].model_copy(update={"unite_cible": None}))

    return out
