# Data

SOTA AI models and datasets can be exported to JSON files.

To export data of a given area, locate in the root of this repo and
run:

```
python data/create_jsons.py <area>
```

where, area can be one of: `cv`, `nlp`, `rl`, or `neuro`, for instance:

```
python data/create_jsons.py cv
```

This will create a JSON file in `data/output/<area>.json` e.g.
`data/output/cv.json`. The file will contain a list of models and datasets of
the given area, alongside its metadata in an standardized format.
