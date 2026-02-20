# Supporting Materials

This folder contains supplementary files from literature papers and other resources.

## Files

### From Paper 1: Chemical Reaction Enumeration (2011)
**Paper:** https://pubs.acs.org/doi/10.1021/ci200379p

- **ci200379p_si_003.xls** - Complete reaction database
  - Contains 7 major reaction classes with SMARTS patterns
  - 1165 N-acylation to amide examples
  - **Key for our project:** {Schotten-Baumann_amide} reaction
  - Will be explored on 2026-01-23 to extract exact SMARTS

- **ci200379p_si_002.pdf** - Visual reaction schemes
  - Graphical representations of all reactions
  - Helpful for understanding reaction mechanisms

## Usage Notes

### ci200379p_si_003.xls
This file contains the Reaction SMARTS patterns we need for peptide coupling.

**Important reactions:**
- N-acylation to amide (1165 examples)
- Schotten-Baumann amide formation (our target)

**Next steps (2026-01-23):**
1. Open and explore the XLS file
2. Locate {Schotten-Baumann_amide} reaction
3. Extract exact SMARTS pattern
4. Test with RDKit
5. Adapt for N-methylated amino acids if needed

---

**Note:** These files are for research purposes. See original paper for full context and citations.
