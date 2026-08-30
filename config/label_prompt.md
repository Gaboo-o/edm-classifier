# EDM weak-labeling prompt — V2

You are assigning weak-supervision genre labels to individual music tracks for
an EDM audio classifier.

Each input record contains:
- `candidate_id`
- artist and track title
- Last.fm `top_tags`
- `retrieval_evidence`
- `label_options`: the relevant taxonomy family/families for this candidate

Return exactly one JSON object per input record.

## Core task

Assign zero, one, or at most two taxonomy labels that are genuinely supported
for the **specific track**.

Prefer the most specific supported child/subgenre. Do not output both a child
and its ancestor. Broad parent labels are allowed when the evidence supports
the family but not a specific child.

This is weak supervision rather than a gold benchmark, so useful high-quality
labels are preferred over excessive abstention. However, do not force a leaf
label merely because the track was retrieved while searching for it.

## Evidence hierarchy

Treat retrieval evidence according to its `strength` and `retrieval_type`.

### Strong: `exact_leaf_tag`
The track itself came from Last.fm's top-track results for that leaf tag.
This is meaningful weak-label evidence and may support assigning the retrieved
leaf when it is consistent with the track's tags and/or track-specific
knowledge.

### Medium: `leaf_tag_artist`
The **artist** is associated with the leaf genre, and this is one of that
artist's top tracks. This is useful evidence, but it does not prove that this
particular track belongs to the leaf. Require corroboration from top tags or
track-specific knowledge before assigning the leaf.

### Weak: `parent_fallback`
The track was retrieved only from the broad parent genre because the requested
leaf had inadequate Last.fm coverage. This is candidate sourcing only.
**Never assign the requested leaf from parent-fallback evidence alone.**
The track may receive the broad parent, a sibling leaf, the requested leaf if
independently supported, or no label.

### Weak: `similar_tag`
A related Last.fm tag produced the track. Treat this as candidate sourcing, not
as proof of the target leaf.

## Other evidence

`top_tags` can corroborate or contradict retrieval evidence, but Last.fm tags
can be generic, sparse, noisy, or culturally inconsistent.

You may use confident **track-specific** musical knowledge to supplement the
provided evidence.

Do not infer a track's leaf genre solely because the artist generally makes
that style.

## Boundary behavior

- Prefer a specific child over its parent when the child is supported.
- Maximum two labels.
- Never output redundant ancestor + descendant labels.
- Cross-genre tracks may receive two non-redundant labels when both are
  genuinely supported.
- If only the broad family is clear, label the parent.
- If the evidence conflicts materially or is insufficient, use `uncertain`.
- If the candidate clearly does not fit the provided EDM taxonomy context,
  use `reject`.

## Confidence

For each assigned label, provide a confidence from 0.0 to 1.0 reflecting the
strength of the track-specific evidence.

As a guide:
- 0.90–1.00: exceptionally clear
- 0.75–0.89: strong
- 0.60–0.74: plausible but not definitive
- below 0.60: generally prefer `uncertain` rather than a forced label

## Output format

Return only JSON matching this structure:

{
  "candidate_id": "<same id as input>",
  "status": "labeled" | "uncertain" | "reject",
  "labels": [
    {
      "id": "<taxonomy id>",
      "confidence": 0.0
    }
  ],
  "reason": "<brief evidence-based explanation>"
}

Rules:
- `labels` contains 0–2 entries.
- `status="labeled"` requires at least one label.
- `status="uncertain"` or `"reject"` may use an empty `labels` list.
- Use only IDs present in `label_options`.
- Do not add prose outside the JSON object.
