# TUKU-v2 First-Batch Candidate Research Report

**Status:** Discovery / research only — **no images downloaded or acquired**.  
**Dataset root:** `TUKU_DATASET_V2`  
**Candidate list:** `metadata/candidate_sources.csv`  
**Split:** all rows `UNASSIGNED` · **Status:** all rows `CANDIDATE`

This report documents why each source/category was selected, licensing status, mask availability, and annotation expectations. Fields marked `UNKNOWN` in the CSV were not fabricated.

---

## Scope and method

| Item | Policy |
|------|--------|
| Image download | **None** |
| COD10K Official Test | **Not searched / not used** |
| Production models | **Untouched** |
| Preferred sources | Wikimedia Commons (per-file CC/PD), USGS PD, NOAA gov works |
| Masks | **Not claimed** unless verified (none verified in this pass) |
| Fabrication | Unknown → `UNKNOWN` |

Primary discovery hubs:

- [Category:Animal camouflage](https://commons.wikimedia.org/wiki/Category:Animal_camouflage) and crypsis subcategories
- USGS public-domain seabed / river imagery metadata
- NOAA OMAO image licensing policy (per-image verification still required)

---

## A. Foliage / vegetation camouflage

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-001–003 | Leaf / foliage crypsis (leaf insect, butterfly) | Insects; SE Asia / India / Switzerland forest | CC BY-SA 2.0–4.0 **clear** | Not available → **manual annotation** |
| CAND-004–005 | Green-on-green frog; iguana in branches | Amphibian + reptile; pond / wetland | CC BY-SA 4.0 **clear** | Manual |
| CAND-006 | Leaf-litter toad | Amphibian; woodland litter | **Public Domain (USGS/NBII)** | Manual |

**Why useful:** Classic crypsis where body outline merges with leaves/vegetation.  
**Hardness:** mostly HARD (CAND-006 MODERATE pending visual QA).

---

## B. Branch / bark camouflage

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-007–008 | Mossy leaf-tailed gecko bark/dermal-flap series | Reptile; Madagascar forest; **EXTREME** | CC BY-SA 4.0 **clear** | Manual; keep `GRP-GECKO-URO-001` together |
| CAND-009 | Stick-like orthopteran among stems | Insect / Orthoptera; Italy woodland | CC BY 4.0 **clear** | Manual |
| CAND-010 | Tawny frogmouth bark roost | Bird; Australia | **MANUAL_LICENSE_REVIEW** (`Copyrighted free use`) | Manual if cleared |
| CAND-011 | Lizard on rock (listed under bark/rock planning) | Reptile; India rocky terrain | CC BY-SA 4.0 **clear** | Manual |

**Why useful:** Disrupted boundaries and texture match to bark/branch — core hard-COD failure modes.

---

## C. Rock / sand / mud camouflage

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-012–014 | Stonefish / reef stonefish rock mimics | Fish; coral reef Egypt / Australia | CC BY-SA 2.0–4.0 **clear** | Manual |
| CAND-015 | Peacock flounder sand burial (multi-frame) | Fish; Hawaii ocean; **EXTREME** | CC BY 2.5 **clear** | Manual; group `GRP-FLOUNDER-KONA-001` |
| CAND-016 | Broadclub cuttlefish sand match | Cephalopod; East Timor | CC BY-SA 3.0 **clear** | Manual |

**Why useful:** Substrate matching and burial — high false-negative risk for detectors.

---

## D. Underwater camouflage

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-017 | Angelshark in seagrass/sand | Fish (elasmobranch); seabed | CC BY 4.0 (iNaturalist-reviewed) **clear** | Manual |
| CAND-018–019 | Octopus coral / Red Sea series | Cephalopod; reef / ocean | CC BY / CC BY-SA 4.0 **clear** | Manual; group `GRP-OCTO-REDSEA-001` |
| CAND-020 | Decorator crab with sponge | Crustacean; California | CC BY 2.0 **clear** | Manual |
| CAND-021–022 | Chameleon prawn on algae | Crustacean; France coast | CC BY-SA 4.0 (re-verify page) | Manual; group `GRP-HIPPOLYTE-WIM-001` |

**Why useful:** Color/texture matching underwater plus clutter — distinct from terrestrial COD10K-heavy foliage.

**Note:** CAND-035 is a **NOAA policy portal**, not a single image. Use only to select future stills after per-image copyright check.

---

## E. Shadow / low-light

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-023–025 | Caiman / crocodile low-contrast water-margin candidates | Reptile; river / wetland / mangrove | **UNKNOWN → MANUAL_LICENSE_REVIEW** | Manual if cleared |
| CAND-026 | Mammal crypsis **category portal** | Mammals (future pick) | Per-file later | Per-file |

**Why useful:** Fill mammal/reptile low-light gap; do **not** acquire until licenses verified.  
**Risk:** CAND-025 is B&W — may be weak for color COD; visual review required.

---

## F. Low-contrast / texture

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-027 | Extreme tiny jungle frog (stick-tip scale) | Amphibian; Brazil jungle; **EXTREME / TINY** | CC BY-SA 3.0 **clear** | Manual |
| CAND-028–030 | Chiton / moth / cowry texture candidates | Mollusc / insect / gastropod | **MANUAL_LICENSE_REVIEW** | Manual if cleared |
| CAND-031 | Arachnid crypsis **category portal** | Arachnids (future) | Per-file later | Per-file |

**Why useful:** Texture / low-contrast / tiny-object regimes poorly covered by easy wildlife photos.

---

## G. Partial occlusion / tiny / broken silhouette

| IDs | Relevance | Diversity add | License | Mask |
|-----|-----------|---------------|---------|------|
| CAND-032 | Annotated highlight of CAND-027 | Same frog group | CC BY-SA 3.0 **clear** | Prefer unmarked original for training |
| CAND-033–034 | Bird / debris-mimic **category portals** | Birds / debris mimics | Per-file later | Per-file |
| CAND-035 | NOAA still selection portal | Underwater animals (future) | Per-image NOAA policy | Manual |

**Group rule:** `GRP-FROG-XIXUAU-001` (CAND-027 + CAND-032) must stay same split later; do not train on annotated overlay if it leaks labels.

---

## H. Hard negatives

| IDs | Hard-negative class | Why deceptive | License | Mask |
|-----|---------------------|---------------|---------|------|
| CAND-036–037 | vegetation-only / empty jungle | Dense foliage false-positive risk | CC BY-SA 4.0 **clear** | **All-zero required** |
| CAND-038–040 | mangrove roots / mangrove clutter | Branch-like / animal-like textures | PD / CC BY-SA / PD | All-zero; CAND-039 creator review |
| CAND-041 | rocks-only underwater | Rock distractors + laser dots | USGS PD **clear** | All-zero |
| CAND-042–043 | sand/mud / empty underwater sets | Substrate textures; sequence groups | USGS PD **clear** | All-zero; pick animal-free frames |
| CAND-044 | grayscale MOUSS empty frames only | Water texture FP risk | NOAA PD (verify card) | All-zero; **never** fish-positive as negative |

**Policy:** Hard negatives must be visually difficult (not random scenic shots). Confirm absence of animals before acquisition.

---

## Licensing summary

### Clear / documented (OK to prioritize for later acquisition)

Most Wikimedia Commons CC BY / CC BY-SA files with verified tags (CAND-001–009, 011–022, 027, 032, 036–038, 040) plus USGS PD (CAND-006, 041–043) and NOAA PD mangrove (CAND-038).

### Manual license review required before any download

| ID | Reason |
|----|--------|
| CAND-010 | `Copyrighted free use`; incomplete machine-readable provenance |
| CAND-023–025 | License/creator not verified this pass |
| CAND-026, 031, 033–034 | Category portals — per-file review later |
| CAND-028–030 | License not verified this pass |
| CAND-035 | Policy portal — per-image NOAA exceptions |
| CAND-039 | Creator attribution incomplete |
| CAND-044 | Verify Hugging Face / NOAA card + grayscale domain fit |

**Do not silently assume permission** for any `UNKNOWN` license.

---

## Mask / annotation expectations

| Claim | Status |
|-------|--------|
| Pre-existing segmentation masks verified for any candidate | **None** |
| Manual binary mask annotation | **Required for all acquired animal images** |
| Hard-negative masks | **All-zero** after visual confirmation of no animal |
| MOUSS segmentation dataset | Exists on HF but **not verified** here; treat as unverified |

---

## Group / leakage notes (no split assignment yet)

| Group | Members | Rule |
|-------|---------|------|
| GRP-GECKO-URO-001 | CAND-007, 008 | Same session → same future split |
| GRP-FLOUNDER-KONA-001 | CAND-015 (+ series frames) | Sequence near-dups |
| GRP-OCTO-REDSEA-001 | CAND-019 (+ series) | Sequence near-dups |
| GRP-HIPPOLYTE-WIM-001 | CAND-021, 022 | Burst/series |
| GRP-FROG-XIXUAU-001 | CAND-027, 032 | Original + annotated aid |
| GRP-HN-USGS-* / GRP-HN-MOUSS | Source sets | Keep related frames together |

`split = UNASSIGNED` for all rows.

---

## Diversity snapshot (planning)

- **Taxa represented or portaled:** insects, reptiles, amphibians, fish, cephalopods, crustaceans, birds (1 + portal), mammals (portal), arachnids (portal), molluscs  
- **Environments:** forest, jungle, woodland, wetland, pond, coral reef, ocean, seabed, mangrove, river, rocky terrain, urban-edge  
- **Gaps to fill next:** more mammals, more arachnids, more snow/ice, more shadow-only terrestrial scenes — only with clear licenses

---

## What this step did **not** do

- No image/dataset downloads  
- No scraping of binary assets  
- No copies into `train/` / `val/` / `hard_negative/`  
- No annotation or training  
- No COD10K Test access  
- No model/checkpoint/config changes  

---

## Exact next step (after human approval)

1. Human review of `candidate_sources.csv`  
2. Clear or reject `MANUAL_LICENSE_REVIEW` rows  
3. Only then begin **manual acquisition** of cleared candidates into staging (still outside production Test)  
4. Annotate → QA → duplicate/group checks → later split assignment  

**STOP — do not download in this step.**
