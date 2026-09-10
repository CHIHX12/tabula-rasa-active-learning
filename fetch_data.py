"""Fetch the five public benchmark datasets into the layout the code expects.

The primary (Ni-Fe-Co-Ce)Ox dataset is not downloadable programmatically: it is
supplementary information to Haber et al., Energy Environ. Sci. 7, 682-688
(2014), doi:10.1039/C3EE43683G, and must be obtained from the publisher.  See
DATASETS.md for the required column names.
"""
import urllib.request, zipfile, io, os
from pathlib import Path

OUT = Path(__file__).parent / "data_external"
OUT.mkdir(exist_ok=True)
UCI = "https://archive.ics.uci.edu/static/public"
JOBS = {
    "concrete.csv":   (f"{UCI}/165/concrete+compressive+strength.zip", None),
    "slump.csv":      (f"{UCI}/182/concrete+slump+test.zip", None),
    "wine_red.csv":   (f"{UCI}/186/wine+quality.zip", "winequality-red.csv"),
    "energy.csv":     (f"{UCI}/242/energy+efficiency.zip", None),
}
for name, (url, member) in JOBS.items():
    dst = OUT / name
    if dst.exists():
        print(f"  have {name}"); continue
    try:
        raw = urllib.request.urlopen(url, timeout=90).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        pick = member or [n for n in z.namelist()
                          if n.lower().endswith((".csv", ".xls", ".xlsx"))][0]
        dst.write_bytes(z.read(pick))
        print(f"  fetched {name} <- {pick}")
    except Exception as e:
        print(f"  could not fetch {name}: {e}")
print("\nsteel_strength.csv: install matminer and run\n"
      "    from matminer.datasets import load_dataset; load_dataset('matbench_steels')\n"
      "OER_database.csv: see DATASETS.md")
