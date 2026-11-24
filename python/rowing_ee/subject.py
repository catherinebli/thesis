"""Participant anthropometrics and resting power (mirror of firmware/lib/core/Subject.h)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

KCAL_PER_DAY_TO_W = 4184.0 / 86400.0   # = 0.0484259...


@dataclass(frozen=True)
class Subject:
    id: str
    mass_kg: float
    height_m: float
    age: float
    sex: str  # "M" or "F"

    def bmr_watts(self) -> float:
        """Mifflin-St Jeor basal metabolic rate, converted from kcal/day to W."""
        s = -161.0 if self.sex.upper().startswith("F") else 5.0
        return (10.0 * self.mass_kg + 625.0 * self.height_m - 5.0 * self.age + s) * KCAL_PER_DAY_TO_W

    def rest_watts(self, k_rest: float) -> float:
        """Resting power used as the estimate floor and for pauses."""
        return self.bmr_watts() * k_rest

    @classmethod
    def from_dict(cls, d: dict) -> "Subject":
        return cls(
            id=str(d.get("id", "S00")),
            mass_kg=float(d["mass_kg"]),
            height_m=float(d["height_m"]),
            age=float(d["age"]),
            sex=str(d.get("sex", "M")).upper()[:1],
        )


def parse_subject_cfg(text: str, defaults: Subject) -> Subject:
    """Parse the key=value subject.cfg format the firmware reads from the SD card.

    Unknown keys and malformed lines are ignored; missing keys keep the default.
    """
    d = {"id": defaults.id, "mass_kg": defaults.mass_kg, "height_m": defaults.height_m,
         "age": defaults.age, "sex": defaults.sex}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if k in ("mass_kg", "height_m", "age"):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        elif k == "id" and v:
            d["id"] = v[:15]
        elif k == "sex" and v:
            d["sex"] = v[:1].upper()
    return Subject.from_dict(d)


def load_subjects(path: Path | str) -> dict[str, Subject]:
    """Read data/subjects.yaml: {S01: {mass_kg: .., height_m: .., age: .., sex: ..}, ...}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {sid: Subject.from_dict({"id": sid, **vals}) for sid, vals in raw.items()}
