# Datasets are not redistributed here

This archive deliberately contains **no** primary data files. All six datasets
are third-party resources whose redistribution terms are set by their original
publishers, so the archive ships the loading and preprocessing code and the
exact provenance instead. `fetch_data.py` places each file where the code
expects it.

| dataset | N | samples | expected path | source |
|---|---|---|---|---|
| (Ni–Fe–Co–Ce)Ox oxygen evolution | 4 | 6074 | `data/OER_database.csv` | Haber, Cai, Jung, Xiang, Mitrovic, Jin, Bell & Gregoire, *Energy Environ. Sci.* **7**, 682–688 (2014), doi:10.1039/C3EE43683G — obtain from the publisher's supplementary information |
| Steel alloy yield strength | 13 | 312 | `data_external/steel_strength.csv` | Matbench `matbench_steels`, doi:10.1038/s41524-020-00406-3 |
| Concrete compressive strength | 8 | 1030 | `data_external/concrete.csv` | UCI ML Repository, Yeh (1998), doi:10.24432/C5PK67 |
| Concrete slump flow | 7 | 103 | `data_external/slump.csv` | UCI ML Repository, Yeh (2007), doi:10.24432/C5FG7D |
| Wine quality (red) | 11 | 1599 | `data_external/wine_red.csv` | UCI ML Repository, Cortez *et al.* (2009), doi:10.24432/C56S3T |
| Building energy efficiency | 8 | 768 | `data_external/energy.csv` | UCI ML Repository, Tsanas & Xifara (2012), doi:10.24432/C51307 |

Required columns are listed in `code/src/data/loader.py` (OER) and in the
`DATASETS` dictionary of `code/generalize_multi.py` (the other five).

Every precomputed trajectory in `precomputed_results/` and
`revision_results/` was produced from these sources, so all reported figures
and tables can be regenerated without the raw data; the raw data is needed only
to re-run the campaigns from scratch.
