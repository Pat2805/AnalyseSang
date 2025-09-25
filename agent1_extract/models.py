# models.py
from __future__ import annotations
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re
from datetime import date

ValType = Union[float, str]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

class BilanResult(BaseModel):
    """
    Objet 'résultat' standardisé pour une valeur de bilan sanguin,
    y compris antécédents. Compatible avec extraction LLM + pipeline.
    """
    model_config = ConfigDict(extra="forbid")  # refuse tout champ inconnu

    # Champs obligatoires
    date_prelevement: str = Field(..., description="YYYY-MM-DD")
    marqueur_key: str = Field(..., description="clé normalisée: ldl, hdl, chol_total, tg, glycemie, creatinine, crp, ...")
    marqueur_label: str = Field(..., description="libellé exactement tel qu'imprimé dans le PDF")
    valeur: ValType = Field(..., description='nombre décimal ("." comme séparateur) ou texte ("positif"/"negatif")')
    unite_source: Optional[str] = Field(None, description='unité telle que dans le PDF (peut être null pour qualitatif)')
    unite_cible: Optional[str] = Field(None, description="unité cible (peut être null pour qualitatif)")
    labo: str = Field(..., description='nom du labo ou "inconnu"')
    pdf_filename: str
    page: int = Field(..., ge=1, description="numéro de page (1-based)")
    from_antecedent: bool = Field(..., description="true si issu d'Antécédents, sinon false")

    # Champs optionnels
    provenance: Optional[str] = Field(None, description='ex: "table:Lipides", "texte:Antécédents", ou null')
    reference_min: Optional[float] = Field(None, description="borne basse extraite du PDF, si numérique")
    reference_max: Optional[float] = Field(None, description="borne haute extraite du PDF, si numérique")

    # --- Validations légères ---
    @field_validator("date_prelevement")
    @classmethod
    def _date_fmt(cls, v: str) -> str:
        if not _DATE_RE.match(v):
            raise ValueError("date_prelevement doit être au format YYYY-MM-DD")
        # sanity check (exclut 2022-13-40)
        year, month, day = map(int, v.split("-"))
        date(year, month, day)  # ValueError si invalide
        return v

    @field_validator("valeur")
    @classmethod
    def _normalize_valeur(cls, v: ValType) -> ValType:
        # Autorise float OU chaîne (pour "positif"/"negatif")
        if isinstance(v, str):
            # harmonise virgule -> point si la chaîne est numérique
            s = v.strip().replace(",", ".")
            try:
                return float(s)
            except ValueError:
                # valeur qualitative, on garde la chaîne
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
