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
`data/output/cv.json`. The file will contain the following keys:

- `datasets`: a list of datastes of the given area with its metadata
- `models`: a list of models of the given area with its metadata
- `sources`: a list of sources (libraries) of the given area with its metadata

For each dataset or model in the lists, the following structure must be met:

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

For the sources the following structure must be met:

```
{
  "name": "source name as it is identified internally",
  "original_name": "source original name",
  "url": "source website url"
}
```

Take this CV source as an example:

```
{
  "name": "torch",
  "original_name": "PyTorch",
  "url": "https://pytorch.org/"
}
```

## Development

To create these JSONs for a given area, each sotaai area subdir must have:

- Model and Dataset abstraction classes must have a `to_dict` instance method.
- The init file of the area e.g. `sotaai/cv/__init__.py` must have the
  `create_models_dict` and `create_datasets_dict` functions.
- The utils file of the area e.g. `sotaai/cv/utils.py` which must have the
  following functions:
  - `map_name_sources`: returns a map of all models or datasets and its sources
  - `map_source_metadata`: returns a map of all sources names and its metadata
- Each wrapper file must have the following variables defined:
  `SOURCE_METADATA`, `MODELS`, and `DATASETS`

If any of the above does not exists or fail, the JSON creation will fail with a
NotImplementedError.
