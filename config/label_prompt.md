# EDM Candidate Weak-Labeling Rules v0.2.0

## Role

You are constructing weak labels for a supervised audio-classification dataset. Classify each **specific recording** into the supplied EDM taxonomy.

This is a **weak-supervision** task, not a gold-standard musicology annotation task. The candidate records were deliberately retrieved from genre-tag charts, so their discovery labels are meaningful noisy evidence. Your job is to verify, refine, combine, or reject those weak labels—not to demand proof beyond reasonable doubt.

Make a useful classification whenever the available evidence reasonably supports one. Use `uncertain` only for genuinely ambiguous, contradictory, or insufficient cases.

## Evidence semantics

Each candidate may contain:

- `discovered_for`: the track appeared in a Last.fm top-track chart for that genre/tag. Treat this as **strong weak evidence**, not ground truth. A higher rank is stronger evidence than a lower rank. Multiple compatible discovery labels reinforce one another.
- `top_tags`: community-applied tags for the specific track. Exact taxonomy labels and clear aliases are strong corroborating evidence. Related genre-family tags are useful secondary evidence. Generic tags such as `electronic`, `dance`, decade tags, countries, moods, artist names, `seen live`, or fandom terms are weak evidence.
- artist/title/MBID: identify the recording. You may use reliable knowledge you already have about the **specific recording** as supplemental evidence. Do not classify a track solely from the artist's general reputation.

The `discovered_for` field is intentionally supplied because it is useful weak supervision. Do not ignore it merely because it is noisy.

## Classification rules

1. Classify the **specific recording**, not merely the artist's usual style.
2. Use only IDs that exist in the supplied taxonomy.
3. Prefer the most specific supported label.
4. A broad/root label is valid when the genre family is clear but a child subtype is not sufficiently supported.
5. If a child label is accepted, do not also output its parent for the same classification fact. Parent labels are derived later.
6. Multi-label classification is allowed when two taxonomy genres genuinely apply. Accept at most two labels.
7. Use `discovered_for` as positive genre evidence. Exact or near-exact discovery labels can justify an accepted weak label when there is no meaningful contradiction, especially at high rank.
8. Use `top_tags` to corroborate, refine, or contradict discovery evidence. An exact matching track tag is particularly strong support.
9. Multiple compatible signals should increase confidence. Examples:
   - discovered for `future_bass` + top tag `future bass` -> very strong `future_bass` support.
   - discovered for `melodic_dubstep` + top tags `dubstep`, `melodic dubstep` -> very strong `melodic_dubstep` support.
   - discovered independently for both `future_bass` and `melodic_dubstep`, with compatible tags -> both labels may be valid.
10. A discovery label without matching top tags can still be accepted when it is plausible, reasonably ranked, consistent with other evidence, and not contradicted.
11. You may use reliable knowledge of the specific track to resolve a boundary or reject an obviously noisy Last.fm tag, but do not invent facts or browse unless explicitly given browsing access.
12. Do not infer a genre solely because the artist commonly produces that genre.
13. BPM, when present, is a soft consistency check only.
14. Apply taxonomy aliases, characteristics, hierarchy, and boundary rules.
15. Do not turn generic descriptors such as `emotional`, `dark`, `melodic`, `energetic`, or `atmospheric` into genre labels unless they occur as part of an actual taxonomy genre and the genre itself is otherwise supported.
16. Use `uncertain` only when no label reaches the acceptance threshold because evidence is genuinely weak, contradictory, or evenly split across an unresolved boundary.
17. Use `out_of_scope` when the recording is clearly not represented by the supplied taxonomy.

## Confidence

Confidence is a heuristic weak-label support score, not a calibrated probability.

- `0.95–1.00`: overwhelming support; exact discovery/tag agreement or several independent compatible signals.
- `0.90–0.94`: very strong support; clear genre assignment with little meaningful ambiguity.
- `0.85–0.89`: strong enough for an accepted weak label; discovery evidence is credible and consistent, even if track tags are incomplete.
- `0.70–0.84`: plausible label that should be retained as a review candidate rather than accepted automatically.
- `0.50–0.69`: weak but potentially useful candidate.
- `<0.50`: omit.

Only labels with confidence >= 0.85 may appear in `labels`.

Do **not** artificially cap confidence below 0.85 merely because the evidence is weak supervision. The entire dataset is weakly supervised. If the evidence strongly supports a genre in this context, assign an accepted label.

## Discovery-rank guidance

Rank is supportive rather than decisive. As a rough heuristic:

- rank 1–10 for an exact taxonomy genre: strong evidence;
- rank 11–20: meaningful evidence;
- rank 21–35: useful but weaker evidence;
- the same track appearing under multiple compatible genre queries substantially strengthens the case;
- conflicting specific top tags can override a misleading discovery result.

Do not mechanically convert rank into confidence; evaluate all supplied evidence together.

## Parent/child examples

- Strong `Melodic Dubstep` evidence -> output `melodic_dubstep`, not both `dubstep` and `melodic_dubstep`.
- Evidence clearly supports Dubstep but not a subtype -> output `dubstep`.
- Strong independent support for both `future_bass` and `melodic_dubstep` -> both may be emitted if each reaches 0.85.
- `melodic` by itself is not evidence for `melodic_dubstep`, `melodic_house`, or `melodic_techno`.
- A track retrieved at rank 4 for `future_bass` with no contradictory specific tags may reasonably receive `future_bass` as an accepted weak label even if its top tags are generic.

## Dataset objective

The goal is to produce a large, useful noisy training set for transfer learning. We will later:

- filter by confidence,
- inspect uncertain cases,
- use artist-separated train/validation/test splits,
- and evaluate the resulting audio classifier independently.

Therefore, avoid both extremes:

- do not blindly copy discovery labels;
- do not abstain whenever evidence is less than perfect.

## Processing discipline

- Classify every input record exactly once.
- Preserve every `candidate_id` exactly.
- Do not silently omit difficult tracks; use `uncertain` instead.
- Do not add tracks that were not supplied.
- When processing a large job, continue until **all records** have been classified.
