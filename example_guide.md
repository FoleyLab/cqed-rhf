# Energy Scan vs theta and phi using CQED-RHF
- /qed-rhf/examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene_field_scan.py
- /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/ORTHO_QEDRHF_SCAN/ortho_field_scan.py
- /Users/jfoley19/Code/cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/META_QEDRHF_SCAN/meta_field_scan.py

# Existing Scan Data

## QED-CCSD:
- Ortho intermediate: /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/ortho_ccsd_energies.txt

- Meta intermediate: /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/meta_ccsd_energies.txt

## QED-RHF
- Nitrobenzene (unsubstituted): /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/nitrobenzene_rhf_energies.txt

# Existing MD Script
- Nitrobenzene (unsubstituted): /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene_tracker.py

# Existing MD Output
- Nitrobenzene (unsubstituted): /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/un_brominated/nitrobenzene.xyz

# Script to generate Exachem QED-CCSD inputs for different values of theta and phi (could easily update with unsubstituted nitrobenzene coordinate)
- Works for ortho, meta, or para inputs: /cqed-rhf/examples/nitrobenzene/ortho_meta_orientation/CCSD_DATA/CCSD_INPUT_GENERATION_SCRIPT/write_exachem_qedcc_input.ipynb
