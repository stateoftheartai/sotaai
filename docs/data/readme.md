# Data

## JSONs Creation

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

For each model or dataset stored in the JSON file, the follow structure should
be met:

```json
{
  "_name": "Dataset or model name",
  "_type": "model or dataset",
  "_tasks": [
    "task-n"
  ],
  "_sources": [
    "source-n"
  ]
  "_paper": "The paper name, only for models when data is available",
  ...
    Any other values that depend on the area must be added here (without _)
  ...
}
```

Take this one of CV as an example:

```
{
  "_name": "InceptionResNetV2",
  "_type": "model",
  "_tasks": [
    "classification"
  ],
  "_sources": [
    "keras"
  ]
  "_paper": null,
  "input_type": "numpy.ndarray",
  "input_shape_height": null,
  "input_shape_width": null,
  "input_shape_channels": 3,
  "input_shape_min_height": 75,
  "input_shape_min_width": 75,
  "output_shape": [
    1536
  ],
  "num_layers": 244,
  "num_params": 54336736,
}
```

## Development

To create these JSONs for a given area, each sotaai area subdir must have:

- Model and Dataset abstraction classes must have a `to_dict` instance method.
- The init file of the area e.g. `sotaai/cv/__init__.py` must have the
  `create_models_dict` and `create_datasets_dict` functions.
- The utils file of the area e.g. `sotaai/cv/utils.py` must have the
  `map_name_sources` function.

If any of the above does not exists or fail, the JSON creation will fail with a
NotImplementedError.
