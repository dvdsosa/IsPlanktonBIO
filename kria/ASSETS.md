# Edge assets — where each file actually comes from

`configs/default_config_paper2.yaml` points at several FAISS indices, SQLite
label databases, and DPU-compiled weights. Not all of them live in this
`kria/` folder — some are identical to files already published for the
desktop pipeline, so they aren't duplicated here.

## FAISS indices and SQLite databases — reuse the desktop copies

These are **byte-identical** (verified by MD5 checksum) to what the desktop
pipeline already ships under `../data/`. Download them from Zenodo as
described in the main [README](../README.md#-pre-trained-models-and-database-files)
and point the Kria config at those paths instead — do not expect to find
them in this folder.

| Path in `configs/default_config_paper2.yaml` | Same file as (under `../data/`) |
|---|---|
| `db/st1_p089.bin` | `databases/vector/faiss_index_stage1_pruned_089.bin` |
| `db/st2_p091.bin` | `databases/vector/faiss_index_stage2_opencv_pruned_091.bin` |
| `db/st1_p089.sqlite` | `databases/sql/plankton_db_stage1_pruned_089.sqlite` |
| `db/st2_p091.sqlite` | `databases/sql/plankton_db_stage2_opencv_pruned_091.sqlite` |

## DPU-compiled weights — edge-specific, on Zenodo

`int8_backbone_stg1.xmodel` and `int8_backbone_stg2.xmodel` have no desktop
equivalent (the PC pipeline uses `.pth`, not Vitis AI-compiled `.xmodel`).
They are archived in this repo under `models/` and are also published on the
project's Zenodo record: [10.5281/zenodo.21676347](https://doi.org/10.5281/zenodo.21676347).

---

[⬅ Back to main README](../README.md)
