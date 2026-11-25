# data/

Everything in here except this file and the templates is ignored by git, because
the repo is public. Share recordings through the team Drive instead.

```
data/
  subjects.csv            copy subjects_template.csv and fill in one row per rower
  pieces.csv              tools/record_piece.py adds a row per recorded piece
  raw/S01/
    S01_P1_imu.csv        suit recording (record_piece.py)
    S01_P1_imu.json       recording details (record_piece.py)
    S01_P1_pm5.csv        Concept2 Logbook stroke export for the same piece (you save it)
```

**subjects.csv**: `subject, weight_kg, height_m, age, sex`. Weight and height go into
the model as features, like Slade et al.

**pieces.csv** (one row per piece):

| column | meaning |
|---|---|
| subject, piece | ids, e.g. S01, P1 |
| intensity | label or target, e.g. easy / moderate / hard |
| drag_factor | PM5 drag factor (keep it the same for everyone) |
| imu_file, pm5_file | paths relative to `data/` |
| offset_s | blank normally. Manual sync: the IMU time (s) at which the PM5 timer read 0 |
| ref_segment | blank = thigh. IMU used to find strokes |
| use | 1 = include, 0 = skip (aborted or bad pieces) |
| notes | anything worth remembering |
