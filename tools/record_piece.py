"""Record one erg piece from the suit over USB.

The Teensy must be running firmware/icm20948_mux5. Close the Arduino Serial
Monitor / Plotter first, since they hold the port.

    python tools/record_piece.py S01 P1 --intensity easy --drag 120 --seconds 250

1. Finds the Teensy (or use --port) and tells the sketch to record (r<seconds>).
2. Counts down the still lead-in: sit still at the catch until it says ROW, then
   start rowing. Use Just Row on the PM5, so its timer starts on the first pull.
3. Saves data/raw/S01/S01_P1_imu.csv plus S01_P1_imu.json (recording details).
4. Adds the piece to data/pieces.csv. After the piece, export the PM5 stroke data
   from the Concept2 Logbook and save it as data/raw/S01/S01_P1_pm5.csv.

Ctrl+C stops early; the partial recording is kept and marked aborted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import serial
    from serial.tools import list_ports
except ImportError:                      # pragma: no cover
    raise SystemExit("needs pyserial: pip install pyserial")

ROOT = Path(__file__).resolve().parents[1]
PJRC_VID = 0x16C0                        # Teensy USB vendor id
PIECE_COLS = ["subject", "piece", "intensity", "drag_factor", "imu_file", "pm5_file",
              "offset_s", "ref_segment", "use", "notes"]
REC_MAX_S = 600                          # icm20948_mux5.ino limit


def find_port() -> str:
    ports = [p for p in list_ports.comports() if p.vid == PJRC_VID]
    if len(ports) == 1:
        return ports[0].device
    if not ports:
        raise SystemExit("No Teensy found. Plug it in, close the Arduino Serial Monitor and "
                         "Plotter, or pass --port.")
    raise SystemExit("Several Teensys found, pass --port: " + ", ".join(p.device for p in ports))


def parse_kv(line: str) -> dict:
    """'#REC_BEGIN,seconds=20,sample_hz=100' -> {'seconds': '20', 'sample_hz': '100'}"""
    out = {}
    for part in line.strip().split(",")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def read_line(ser, deadline: float) -> str | None:
    """Next decoded line, or None once the deadline passes with nothing received."""
    while time.time() < deadline:
        raw = ser.readline()
        if raw:
            return raw.decode("ascii", errors="replace").strip()
    return None


def add_piece(data: Path, row: dict, overwrite: bool):
    path = data / "pieces.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=PIECE_COLS)
    hit = (df.get("subject", pd.Series(dtype=str)) == row["subject"]) & \
          (df.get("piece", pd.Series(dtype=str)) == row["piece"])
    if hit.any():
        if not overwrite:
            print(f"note: {row['subject']} {row['piece']} is already in pieces.csv, left unchanged")
            return
        df = df[~hit]
    df = pd.concat([df, pd.DataFrame([row]).astype(str)], ignore_index=True)
    cols = PIECE_COLS + [c for c in df.columns if c not in PIECE_COLS]
    df[cols].to_csv(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="subject id, e.g. S01 (must be in data/subjects.csv)")
    ap.add_argument("piece", help="piece id, e.g. P1")
    ap.add_argument("--intensity", default="", help="e.g. easy / moderate / hard or a target split")
    ap.add_argument("--drag", default="", help="PM5 drag factor for this piece")
    ap.add_argument("--seconds", type=int, default=250, help=f"recording length incl. lead-in (max {REC_MAX_S})")
    ap.add_argument("--lead-in", type=int, default=10, help="seconds to sit still before rowing")
    ap.add_argument("--port", help="serial port (default: the one Teensy that is plugged in)")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--notes", default="")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing recording of this piece")
    args = ap.parse_args(argv)

    if not 1 <= args.seconds <= REC_MAX_S:
        raise SystemExit(f"--seconds must be 1..{REC_MAX_S} (the sketch's limit)")
    subj, piece = args.subject.strip(), args.piece.strip()
    rawdir = args.data / "raw" / subj
    imu_path = rawdir / f"{subj}_{piece}_imu.csv"
    meta_path = imu_path.with_suffix(".json")
    pm5_rel = f"raw/{subj}/{subj}_{piece}_pm5.csv"
    if imu_path.exists() and not args.overwrite:
        raise SystemExit(f"{imu_path} exists. Use another piece id or --overwrite.")
    subjects = args.data / "subjects.csv"
    if not subjects.exists() or subj not in pd.read_csv(subjects, dtype=str).get("subject", pd.Series(dtype=str)).values:
        print(f"note: {subj} is not in {subjects} yet; add their weight and height before building the dataset")
    rawdir.mkdir(parents=True, exist_ok=True)

    port = args.port or find_port()
    started = datetime.now().isoformat(timespec="seconds")
    begin, end, rows, aborted = {}, {}, 0, False
    nan_frames: dict[str, int] = {}
    tmp = imu_path.with_suffix(".csv.part")
    with serial.Serial(port, 115200, timeout=0.2) as ser, open(tmp, "w", newline="") as f:
        time.sleep(0.3)
        ser.reset_input_buffer()
        ser.write(f"r{args.seconds}\n".encode())
        deadline = time.time() + 5
        while True:                                   # skip plotter output until the header
            line = read_line(ser, deadline)
            if line is None:
                raise SystemExit("The Teensy did not start recording. Is icm20948_mux5 uploaded, "
                                 "and is the Arduino Serial Monitor closed?")
            if line.startswith("#REC_BEGIN"):
                begin = parse_kv(line)
                break
        header = read_line(ser, time.time() + 2)
        if not header or not header.startswith("sample,t_s"):
            raise SystemExit(f"unexpected header from the sketch: {header!r}")
        f.write(header + "\n")
        cols = header.split(",")
        seg_idx = {c[:-len("_ax_mps2")]: i for i, c in enumerate(cols) if c.endswith("_ax_mps2")}
        nan_frames = {s: 0 for s in seg_idx}
        hz = float(begin.get("sample_hz", 100))
        total = int(begin.get("seconds", args.seconds))
        print(f"Recording {total} s from {port} -> {imu_path}")
        print(f"Sit still at the catch. Start rowing (Just Row) when it says ROW.")
        shown, rowed = -1, False
        try:
            while True:
                line = read_line(ser, time.time() + 3)
                if line is None:
                    raise SystemExit("No data from the Teensy for 3 s; the partial file is kept as "
                                     f"{tmp}")
                if line.startswith("#REC_END"):
                    end = parse_kv(line)
                    break
                if not line or line.startswith("#"):
                    continue
                f.write(line + "\n")
                rows += 1
                parts = line.split(",")
                for s, i in seg_idx.items():
                    if i < len(parts) and parts[i].strip().lower() == "nan":
                        nan_frames[s] += 1
                t = rows / hz
                sec = int(t)
                if sec != shown:
                    shown = sec
                    if t < args.lead_in:
                        print(f"  still... {args.lead_in - sec}", flush=True)
                    elif not rowed:
                        print("  ROW!", flush=True)
                        rowed = True
                    elif sec % 15 == 0:
                        bad = {s: n for s, n in nan_frames.items() if n}
                        print(f"  {sec:4d} s of {total}" + (f"   failed reads: {bad}" if bad else ""), flush=True)
        except KeyboardInterrupt:
            aborted = True
            print("\nStopping...")
            ser.write(b"x\n")
            deadline = time.time() + 3
            while (line := read_line(ser, deadline)) is not None:
                if line.startswith("#REC_END"):
                    end = parse_kv(line)
                    break
                if line and not line.startswith("#"):
                    f.write(line + "\n")
                    rows += 1
    tmp.replace(imu_path)

    meta = dict(subject=subj, piece=piece, intensity=args.intensity, drag_factor=args.drag,
                notes=args.notes, started=started, port=port, lead_in_s=args.lead_in,
                rows=rows, aborted=aborted or end.get("aborted") == "1", failed_reads=nan_frames,
                rec_begin=begin, rec_end=end, imu_file=imu_path.name, pm5_file=Path(pm5_rel).name,
                firmware="icm20948_mux5")
    meta_path.write_text(json.dumps(meta, indent=2))
    add_piece(args.data, dict(subject=subj, piece=piece, intensity=args.intensity, drag_factor=args.drag,
                              imu_file=f"raw/{subj}/{imu_path.name}", pm5_file=pm5_rel, offset_s="",
                              ref_segment="", use="0" if meta["aborted"] else "1", notes=args.notes),
              args.overwrite)

    late, errs = int(end.get("late", 0) or 0), int(end.get("i2c_errors", 0) or 0)
    print(f"\nSaved {rows} rows ({rows / hz:.0f} s) -> {imu_path}")
    bad = {s: n for s, n in nan_frames.items() if n}
    if bad:
        print(f"Failed reads per IMU (logged as NaN): {bad}")
    if meta["aborted"]:
        print("Stopped early: the piece is in pieces.csv with use=0. Set it to 1 to include it anyway.")
    if late or errs:
        print(f"Check: {late} late samples, {errs} failed I2C reads (see {meta_path.name})")
    print(f"Next: export this piece's stroke data from the Concept2 Logbook and save it as\n"
          f"      {args.data / pm5_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
