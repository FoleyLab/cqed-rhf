# Example Guide: Energy Scans and MD with CQED-RHF

## Energy Scan Scripts (θ and φ grid)
- **Nitrobenzene (unsubstituted):** `examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene_field_scan.py`
- **Ortho-brominated:** `examples/nitrobenzene/ortho_meta_orientation/ORTHO_QEDRHF_SCAN/ortho_field_scan.py`
- **Meta-brominated:** `examples/nitrobenzene/ortho_meta_orientation/META_QEDRHF_SCAN/meta_field_scan.py`

## Existing Scan Data

### QED-CCSD
- **Ortho-brominated:** `examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/ortho_ccsd_energies.txt`
- **Meta-brominated:** `examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/meta_ccsd_energies.txt`

### QED-RHF
- **Nitrobenzene (unsubstituted):** `examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/nitrobenzene_rhf_energies.txt`

> ⚠️ **Note:** QED-CCSD energies for unsubstituted nitrobenzene on the same θ/φ grid as the QED-RHF data are not yet available.

## MD Scripts
- **Nitrobenzene (unsubstituted):** `examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene_tracker.py`

## MD Output
- **Nitrobenzene (unsubstituted):** `examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene.xyz`

## ExaChem QED-CCSD Input Generation
Jupyter notebook to generate ExaChem QED-CCSD inputs for different θ and φ values. Works for ortho, meta, or para species; coordinates can be easily updated for unsubstituted nitrobenzene.
- `examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/CCSD_INPUT_GENERATION_SCRIPT/write_exachem_qedcc_input.ipynb`
