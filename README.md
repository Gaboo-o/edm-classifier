# EDM evidence collector v0.1

This stage gathers evidence only. It does **not** assign taxonomy labels.

## Pipeline position

```text
yt-dlp .info.json / track identity
          |
          v
  evidence_collector.py
          |
          v
 weak-label input JSON
          |
          v
 LLM + taxonomy.yaml + weak_label_prompt.md
          |
          v
 weak-label output JSON
          |
          v
 accepted labels -> training dataset
 uncertain labels -> review queue
```

## Sources

- **MusicBrainz**: recording-level genres/tags. No API key. The collector self-throttles to ~1 request/sec and sends a User-Agent.
- **Last.fm**: track-level community top tags. Optional; set `LASTFM_API_KEY`.
- **Discogs**: release/master genre/style evidence only after the collector finds the target title in that release/master's tracklist. Optional; set `DISCOGS_TOKEN`.
- **yt-dlp sidecar**: source-provided YouTube tags/categories, deliberately marked low reliability.

Artist-level tags are deliberately excluded from v0.1 because they can create label leakage: an artist's usual genre is not proof of a specific recording's genre.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r evidence_collector_requirements.txt
```

Optional credentials:

```bash
export LASTFM_API_KEY="..."
export DISCOGS_TOKEN="..."
```

For MusicBrainz, optionally set a more specific User-Agent containing contact/project information:

```bash
export MUSIC_METADATA_USER_AGENT="ytm-edm-dataset/0.1 (your-contact-or-project-url)"
```

## Collect from an existing yt-dlp sidecar

```bash
python evidence_collector.py \
  --info-json "$HOME/Music/ytm-import/002 - Take You Down [VIDEO_ID].info.json" \
  -o evidence/take_you_down.json
```

The exact filename does not matter; shell tab-completion is recommended.

## Collect from explicit identity metadata

```bash
python evidence_collector.py \
  --track-id VIDEO_ID \
  --youtube-id VIDEO_ID \
  --artist "ILLENIUM" \
  --title "Take You Down" \
  --duration 215 \
  -o evidence/take_you_down.json
```

## Disable individual providers

```bash
python evidence_collector.py ... --no-discogs --no-lastfm
```

## Output

The output validates against `weak_label_input.schema.json` and is ready to be given to the LLM labeling stage together with `taxonomy.yaml` and `weak_label_prompt.md`.
