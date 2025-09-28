from __future__ import annotations
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import date
import re

ValType = Union[float, str]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

class BilanResult(BaseModel):
    """Objet standardisé pour une valeur de bilan sanguin (y compris antécédents)."""
    model_config = ConfigDict(extra="forbid")  # refuse tout champ inconnu

    # obligatoires
    date_prelevement: str = Field(..., description="YYYY-MM-DD")
    marqueur_key: str = Field(..., description="clé normalisée: ldl, hdl, chol_total, tg, glycemie, creatinine, crp, ...")
    marqueur_label: str = Field(..., description="libellé tel que dans le PDF")
    valeur: ValType = Field(..., description='nombre (séparateur ".") ou chaîne ("positif"/"negatif")')
    unite_source: Optional[str] = Field(None, description='unité telle que dans le PDF, ou null')
    unite_cible: Optional[str] = Field(None, description="unité cible selon config, ou null")
    labo: str = Field(..., description='nom labo, sinon "inconnu"')
    pdf_filename: str
    page: int = Field(..., ge=1)
    from_antecedent: bool

    # optionnels
    provenance: Optional[str] = None
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None

    # validations
    @field_validator("date_prelevement")
    @classmethod
    def _date_fmt(cls, v: str) -> str:
        if not _DATE_RE.match(v):
            raise ValueError("date_prelevement doit être au format YYYY-MM-DD")
        y, m, d = map(int, v.split("-"))
        date(y, m, d)  # ValueError si invalide
        return v

    @field_validator("valeur")
    @classmethod
    def _normalize_valeur(cls, v: ValType) -> ValType:
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            try:
                return float(s)
            except ValueError:
                return v.strip().lower()
        return float(v)

    @field_validator("unite_source", "unite_cible")
    @classmethod
    def _normalize_unit(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    @field_validator("labo", "pdf_filename", "marqueur_key", "marqueur_label")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("champ texte obligatoire vide")
        return s
