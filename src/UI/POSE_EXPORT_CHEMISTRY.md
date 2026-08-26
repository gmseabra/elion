# Docked-pose export: bond orders and formal charges

*Why a chromen-4-one opened in Maestro as a 4-hydroxychromene, why a quinoline
N-oxide opened as N-hydroxy, and what now happens instead.*

---

## The report

Two exported poses were drawn wrongly in Maestro:

| | what Maestro drew | what it should be |
|---|---|---|
| **quinoline 1-oxide** | `N–OH` — an extra hydrogen on the oxygen, no charges | `N⁺–O⁻` |
| **4*H*-chromen-4-one** | 4-hydroxychromene — the ketone demoted to an alcohol | the 4-one |

Both were reported as frequent, "especially structures with an oxygen in the
ring". The consequence is not cosmetic: an H-bond donor appears where an
acceptor belongs, and the net charge is wrong.

## The cause

The pipeline used **PDB as the interchange format between every stage**:

```
SMILES ──RDKit──▶ .pdb ──obabel──▶ .mol2 ──obabel──▶ .pdbqt ──Vina──▶ *_out.pdbqt
                                                                         │
                            download button ◀────────────────────────────┘
```

A PDB file has no practical way to record a formal charge, and a PDBQT carries
no bond orders at all. So the chemistry was thrown away at step one and
**re-guessed from 3D coordinates twice afterwards** — once by obabel, once by
the download route.

The download route's guess (`_perceive_pdb_orders` in `pose_routes.py`) had a
specific defect: *any* planar 5-/6-ring of C/N/O/S was given alternating Kekulé
double bonds, **with no valence cap**. Run on a γ-pyranone, that hands the ring
ether oxygen a double bond it cannot have:

```
ring O ends up with THREE bonds  →  Maestro cannot keep an impossible valence
                                 →  it demotes the real C4=O to a single bond
                                 →  that oxygen is now one bond short
                                 →  it caps it with an H  →  4-hydroxychromene
```

The N-oxide fails the other way round: the connectivity was right, but with no
formal charges written anywhere the nitrogen is a bare 4-valent N and the oxygen
is one bond short — so the reader adds an H, and `N⁺–O⁻` becomes `N–OH`.

Reproduced from the code, the same failure appears for coumarin, flavone,
pyranone, chromanone and pyridine N-oxide. Plain ethers and morpholines are
fine, which matches the report exactly.

## What this did **not** affect: the docking

Vina reads AutoDock atom types and coordinates — nothing else. Both chains were
rebuilt and compared atom by atom:

```
PDB → mol2 → pdbqt   (the lossy chain that was used)
SDF → pdbqt          (a clean chain)
```

**Zero atom-type differences, identical hydrogen counts.** The affinities and
poses in the existing results stand; nothing needs re-docking. This was an
export/annotation problem, not a scoring one.

## The fix

### 1. Chemistry is rebuilt from the ligand's own SMILES

New module `uiapp/core/pose_chem.py`. Three tiers, in order:

| tier | what it uses | when |
|---|---|---|
| `template` | the ligand's SMILES, matched onto the docked skeleton; bond orders **and** formal charges copied across | whenever a SMILES can be found — exact by construction |
| `perceived` | RDKit `rdDetermineBonds` (xyz2mol), a real valence-consistent solver, scored against the geometry heuristic and the better answer taken | no SMILES available |
| `geometry` | the old heuristic, now valence-capped and charge-aware | no RDKit at all |

SMILES lookup order, per file: a `<stem>.smi` sidecar → a `REMARK … SMILES`
line in the file → any CSV/SMI table beside the batch or in its parent
directories (`Name`/`ligand_id` + `SMILES` columns), including the pipeline's own
`results_TS/*.csv`. Batch names like `479575_20260806132002_out.pdbqt` are
matched on `479575` as well as on the full stem.

A template that does not match the docked skeleton is **rejected**, not forced —
the export falls through to perception and says so in the manifest. Stereo­
chemistry is taken from the docked coordinates, not from the SMILES, and a
disagreement is reported rather than silently resolved.

### 2. Every batch download now also contains `.sdf`

`files/*.sdf` alongside `files/*.pdb`, plus `_best_poses.sdf`.

**Open the `.sdf` in Maestro.** Bond orders and formal charges are first-class
fields in an MDL molfile, so there is nothing left to perceive and nothing that
can be perceived wrongly. The `.pdb` files are still written — same atoms, same
order, same coordinates — and are much improved (`HETATM` records, symmetric
order-duplicated CONECT, formal charge in columns 79-80), but the SDF is the one
to trust when the drawing matters.

The invariant the tests enforce on the exported PDB: for **every** atom, the
number of CONECT entries equals the element's valence adjusted by the formal
charge in columns 79-80. When that holds, a reader has nothing to repair.

### 3. `_manifest.csv` says where the chemistry came from

New columns: `chemistry` (`template` / `perceived` / `geometry`), `net_charge`,
`smiles`, `smiles_from`, `note`. The UI reports the same counts after a
download, so a batch that fell back to perception is visible immediately instead
of being discovered later in Maestro.

### 4. The upstream pipeline no longer serialises through PDB

* `pipeline/active_learning/get_pdbs_from_results_TS_smiles.py` now writes
  `<name>.sdf` (lossless) and a `<name>.smi` sidecar next to the `.pdb`, and puts
  a `REMARK … SMILES` line in the PDB.
* `engines/vina/pdb_to_pdbqt_cpu_thread.py` converts **straight from the SDF**
  (`obabel -isdf … -opdbqt`), skipping the mol2 round-trip that re-perceived
  everything. The old `.pdb → .mol2 → .pdbqt` route is kept as a fallback for
  batches that predate the `.sdf` files, and is reported separately in the
  summary.

### 5. The 3D viewer had the same bug

`perceivePdbBondOrders` in `web/static/js/pose.js` is the browser twin of the
server heuristic and shared the missing valence cap. It is now the same
algorithm as the Python side — cross-checked bond-for-bond and charge-for-charge
on 14 molecules — so what you see in the app matches what you export.

### 6. Already-downloaded batches: `tools/reexport_poses.py`

```bash
# rewrite an existing folder of *_out.pdbqt as correct .sdf + .pdb
python tools/reexport_poses.py /data/vina/LGBM_suzuki/4/pdbqt \
    --smiles-csv /data/results_TS/LGBM_suzuki_TS_4.csv \
    --out /data/vina/LGBM_suzuki/4/export
```

No re-docking, no coordinate changes — only the chemistry annotation is rebuilt.
`--dry-run` reports what would happen; `--all-poses` keeps every mode instead of
just Vina's best.

## Tests

`tests/test_pose_chem.py` — 103 cases, including the actual reported pose
(`tests/data/479575_20260806132002_out.pdb`, which must come back as
C20H12BrN5O2 with exactly one +1 and one −1):

* all three tiers on 26 molecules spanning ring oxygens, N-oxides, nitro,
  carboxylate, phosphate, amidinium, pyridinium, sulfonyl, tetrazole;
* the per-atom valence invariant on the exported PDB;
* SDF round-trip equality, including net charge;
* coordinates and atom order provably untouched;
* a deliberately wrong template is rejected rather than forced.

```bash
pytest tests/test_pose_chem.py -q
```

---

## 中文摘要（可直接转发）

**问题**：导出的 PDB 在 Maestro 里被画错——喹啉 N-氧化物显示成 N–OH（少了
N⁺/O⁻ 电荷），色酮（chromen-4-one）显示成 4-羟基色烯（酮变成了醇）。氢的位置
不对，导致氢键给体/受体互换，电荷也不对。

**原因**：整条流程用 PDB 作为中间格式。PDB 记不了形式电荷，PDBQT 连键级都没有，
所以化学信息在第一步就丢了，后面 obabel 和下载按钮各自**从三维坐标重新猜**一遍。
下载按钮那段代码把任何平面的五元/六元 C/N/O/S 环都强行交替成双键，**没有做价键
上限检查**，于是环上的醚氧被安上一个双键（三价氧，根本不存在）。Maestro 读到不
可能的价态就会"修"它：把真正的 C4=O 降成单键，再给氧补一个 H —— 就成了
4-羟基色烯。N-氧化物则是电荷没写进文件，氧少一根键，Maestro 同样补了个 H。

**对接结果没有受影响**：Vina 只读 AutoDock 原子类型和坐标。把有问题的链路和干净
链路逐原子对比，原子类型完全一致、氢原子数也一致。**现有 734 个结果的打分和构象
都是有效的，不需要重新对接。**

**修复**：
1. 导出时用配体自己的 SMILES 重建键级和形式电荷（找不到 SMILES 时用 RDKit 的
   价键一致性算法，两种结果打分取优）；
2. 下载包里**新增 `.sdf` 文件** —— 请在 Maestro 里打开 `.sdf`：键级和电荷是显式
   写在文件里的，没有任何需要"猜"的余地。`.pdb` 仍然保留（原子、顺序、坐标完全
   不变），并且补上了 HETATM、对称 CONECT 和第 79-80 列的形式电荷；
3. `_manifest.csv` 里新增 `chemistry` 列，标明每个分子的化学信息来源；
4. 上游脚本改为输出 `.sdf` + `.smi`，`obabel` 直接从 `.sdf` 转 `.pdbqt`，不再经过
   会重新猜结构的 `.mol2`；
5. 网页 3D 视图里同样的 bug 也一并修了；
6. 已经下载过的批次可以用 `tools/reexport_poses.py` 直接重新导出，不用重跑对接。
