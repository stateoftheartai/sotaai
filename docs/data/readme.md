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

To create the JSONs of all areas, do not pass any argument:

```
python data/create_jsons.py
```

This will create a JSON file in `data/output/items.json`. The file will contain
the following keys:

- `datasets`: a list of datastes of the given area(s) with its metadata
- `models`: a list of models of the given area(s) with its metadata
- `sources`: a list of sources (libraries) of the given area(s) with its metadata
- `tasks`: a list of tasks of the given area(s) with its metadata
- `area`: a list of areas with its metadata

For each Dataset or Model of the lists, the following structure must be met:

```json
{
  "name": "Dataset or model name",
  "area": "Area name e.g. cv"
  "type": "model or dataset",
  "sources": [
    "source-n"
  ]
  "tasks": [
    "task-n"
  ],
  "paper": "The paper name, only for models when data is available",

  ...

  Area dependent attributes must be added here prefixed with
  <area>_attribute_name e.g. cv_output_shape

  ...

}
```

This is an example taken from CV:

```
{
  "name": "InceptionResNetV2",
  "area": "cv",
  "type": "model",
  "sources": [
    "keras"
  ]
  "tasks": [
    "classification"
  ],
  "paper": null,
  "cv_input_type": "numpy.ndarray",
  "cv_input_shape_height": null,
  "cv_input_shape_width": null,
  "cv_input_shape_channels": 3,
  "cv_input_shape_min_height": 75,
  "cv_input_shape_min_width": 75,
  "cv_output_shape": [
    1536
  ],
  "cv_num_layers": 244,
  "cv_num_params": 54336736,
}
```

For the sources the following structure must be met:

```
{
  "name": "source identifier",
  "original_name": "source name",
  "url": "source website url"
}
```

This is an example taken from CV:

```
{
  "name": "torch",
  "original_name": "PyTorch",
  "url": "https://pytorch.org/"
}
```

## Development

To create these JSONs of a given area, **each sotaai area directory must have**:

- Model and Dataset abstraction classes must have a `to_dict` instance method.
- The init file of the area e.g. `sotaai/cv/__init__.py` must have the
  `create_models_dict` and `create_datasets_dict` functions.
- The utils file of the area e.g. `sotaai/cv/utils.py` must have the
  following functions:
  - `map_name_sources`: returns a map of all models or datasets and their sources
  - `map_source_metadata`: returns a map of all sources names and their metadata
- Each wrapper file must have the following variables defined:
  `SOURCE_METADATA`, `MODELS`, and `DATASETS`

If any of the above does not exists or fail, the JSON creation will fail with a
**NotImplementedError**.
