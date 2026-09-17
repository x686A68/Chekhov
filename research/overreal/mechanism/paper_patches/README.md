# Paper patches held back for Overleaf

2026-09-17: Overleaf could not merge, so the two commits made after the last
Overleaf sync (7c44b34, "Updates from Overleaf") were reverted with a normal
commit on the paper repo. Their content is saved here as patches and is to
be re-applied once Overleaf has synced.

1. `0001-6.3-drop-the-appearance-paragraph...patch`
   - removes the paragraph "That attention is what puts the target into the image"
   - the cucumber figure is referenced from the end of the first paragraph
   - Table 5 loses its two AUC columns (caption shortened)
   - figure caption without the 15% / 10% numbers
   - third paragraph: the AUC_c sentence replaced by a pointer to the appendix
2. `0002-6.3-and-appendix-no-AUC-anywhere...patch`
   - third paragraph: the "Nor does reading the cue words more help" sentences removed
   - fourth paragraph: "The attention again predicts appearance (AUC ...)" replaced
     by "Table X gives the numbers per family."
   - appendix C.4: AUC columns removed from the per-family table, the detail table
     and the expanded table; AUC clauses removed from the intro paragraph, from
     "Where the families differ" and from the expanded-table description
   - intro: "and that attention puts the target into the image" -> "and renders it"

Re-apply after the Overleaf sync with, in the paper submodule,
`git am research/../paper_patches/*.patch` (or `git revert` of the revert
commit), or by hand if Overleaf's edits touched the same lines.
