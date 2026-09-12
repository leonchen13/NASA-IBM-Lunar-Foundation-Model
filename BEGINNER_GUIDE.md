# Beginner's Guide to the NASA-IBM Lunar Foundation Model

This guide explains what the NASA-IBM Lunar Foundation Model (Lunar-FM) is, what it can do, and how to run the examples in this repository. It assumes you are new to machine learning and remote-sensing models.

## 1. What is Lunar-FM?

Lunar-FM is a machine-learning model trained on different kinds of Moon data. These include visible images, ultraviolet images, elevation, slope, aspect, observation geometry, and static scientific maps.

It is a **foundation model**, which means it learned general patterns from a large collection of lunar data. We can reuse those patterns for more specific jobs instead of training every new model from the beginning.

Lunar-FM is not a chatbot. It accepts lunar images or map layers and produces numerical features. A task-specific component, called a **head**, converts those features into useful outputs such as crater boxes, segmentation masks, or prospectivity maps.

```text
Lunar data -> Lunar-FM backbone -> Task head -> Prediction
```

Examples:

```text
NAC image -> Lunar-FM -> Crater detector -> Crater bounding boxes
NAC image -> Lunar-FM -> Segmenter      -> IMP pixel mask
Map layers -> Lunar-FM -> Regressor      -> Ice-prospectivity map
```

## 2. Important vocabulary

| Term | Beginner-friendly meaning |
|---|---|
| **Model** | A program whose behavior was learned from examples. |
| **Foundation model** | A model trained broadly and reused for several tasks. |
| **Backbone** | The main feature-extraction part of the model. |
| **Task head** | The final part that turns features into a specific prediction. |
| **Checkpoint** | A file containing learned model weights. |
| **Modality** | One kind of input data, such as an image, elevation, or slope. |
| **Inference** | Using a trained model to make predictions. |
| **Fine-tuning** | Continuing training for a new or specialized task. |
| **Detection** | Locating objects with boxes, such as craters. |
| **Segmentation** | Assigning a class to every pixel. |
| **Regression** | Predicting a continuous value for every pixel or sample. |
| **Embedding** | A numeric summary of the patterns in an input. |

## 3. What data can it use?

The released model supports two main resolution families.

### Low-resolution WAC data

| Key | Data | Channels |
|---|---|---:|
| `vis` | Visible reflectance | 5 |
| `uv` | Ultraviolet reflectance | 2 |
| `dtm` | Digital terrain/elevation model | 1 |
| `slope` | Terrain slope | 1 |
| `aspect` | Sine and cosine of terrain aspect | 2 |

### High-resolution NAC data

| Key | Data | Channels |
|---|---|---:|
| `nac` | Panchromatic NAC image | 1 |
| `dtm_3m` | Approximately 3 m terrain/elevation model | 1 |
| `slope_3m` | Approximately 3 m terrain slope | 1 |
| `aspect_3m` | Sine and cosine of terrain aspect | 2 |

The model can also use sequence-like information:

- `metadata`: acquisition geometry such as incidence, emission, and phase angles; sub-solar coordinates; image-center coordinates; and solar ground azimuth.
- `static_maps`: contextual values derived from lunar scientific products.

## 4. What is already installed here?

This workspace currently contains:

- A Python environment in `.venv/`
- The base checkpoint in `backbone/checkpoint.pt`
- The required model configuration in `backbone/config.yaml`
- Nine modality tokenizers in `tokenizers/`
- All four downstream datasets in `data/`
- Published task checkpoints in `checkpoints/downstream/`
- JupyterLab for running notebooks

The downloaded files use roughly 23 GB of storage. Most downloaded artifacts are ignored by Git so they are not accidentally committed.

## 5. Start the environment

Open a terminal in this repository and run:

```bash
source .venv/bin/activate
```

You should see `(.venv)` near the beginning of your terminal prompt.

When you are finished, leave the environment with:

```bash
deactivate
```

## 6. Your first example: IMP segmentation

This is a good first example because all required data and trained weights are installed, and it runs successfully on CPU.

```bash
source .venv/bin/activate

PYTHONPATH=. terratorch test \
  --config checkpoints/downstream/imp/config.yaml \
  --ckpt_path checkpoints/downstream/imp/ni_lfm_ps8_frozen_s44.ckpt \
  --trainer.accelerator=cpu \
  --trainer.devices=1 \
  --data.batch_size=1 \
  --data.num_workers=0
```

This command performs **inference** on the IMP test images and compares the predicted pixel masks with the known labels.

In the verified local run, the model produced approximately:

- IMP-class IoU: `0.583`
- Mean IoU: `0.770`
- Pixel accuracy: `0.960`

IoU means *intersection over union*. It measures how much the predicted region overlaps the labeled region. Higher is better.

You can inspect the supplied comparison image at [IMP predictions](checkpoints/downstream/imp/imp_predictions.png). Green areas are labels and magenta outlines are predictions.

## 7. Use case: crater detection

Crater detection produces a bounding box and confidence score for each detected crater.

Possible applications include:

- Creating an initial crater catalog
- Counting and measuring craters
- Comparing crater density between regions
- Screening large image collections before expert review
- Adding crater hazards to a terrain-planning cost map

### NAC crater evaluation

```bash
PYTHONPATH=. terratorch test \
  --config checkpoints/downstream/craters/NAC_config.yaml \
  --ckpt_path checkpoints/downstream/craters/NAC_ni_lfm_ps8_s44.ckpt \
  --trainer.accelerator=cpu \
  --trainer.devices=1
```

See [NAC crater predictions](checkpoints/downstream/craters/crater_prediction_NAC.png).

### WAC crater evaluation

```bash
PYTHONPATH=. terratorch test \
  --config checkpoints/downstream/craters/WAC_config.yaml \
  --ckpt_path checkpoints/downstream/craters/WAC_ni_lfm_ps8_lora_s46.ckpt \
  --trainer.accelerator=cpu \
  --trainer.devices=1
```

See [WAC crater predictions](checkpoints/downstream/craters/crater_prediction_WAC.png).

The example figures use green boxes for ground truth and blue boxes for predictions.

## 8. Use case: ice-prospectivity mapping

The ice-prospectivity model combines polar terrain and scientific map layers to predict a continuous prospectivity value at every pixel.

Possible applications include:

- Ranking polar regions for more detailed analysis
- Studying how slope, aspect, temperature, and shadow-related products interact
- Comparing results when individual modalities are removed
- Producing a candidate map for expert review

Run it with:

```bash
PYTHONPATH=. terratorch test \
  --config checkpoints/downstream/ice/config.yaml \
  --ckpt_path checkpoints/downstream/ice/ni_lfm_ps8_all_modalities_s42.ckpt \
  --trainer.accelerator=cpu \
  --trainer.devices=1
```

See [ice-prospectivity predictions](checkpoints/downstream/ice/ice_prospectivity_predictions.png).

Important: this output represents a knowledge-driven prospectivity model. It is not a direct measurement or confirmation of water ice.

## 9. Use case: reusable feature extraction

You can use the backbone without a task head to convert a lunar tile into an embedding. Embeddings are useful for image search, clustering, anomaly detection, and small custom classifiers.

```python
import torch
import terratorch_integration
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

model = TERRATORCH_BACKBONE_REGISTRY.build(
    "ni_lfm_v1_base",
    modalities=["vis"],
    cfg="backbone/config.yaml",
    checkpoint_path="backbone/checkpoint.pt",
    remove_register_tokens=True,
).eval()

# Example batch: one 256 x 256 tile with five visible channels.
# Real data must use the correct channel order and normalization.
tile = torch.randn(1, 5, 256, 256)

with torch.inference_mode():
    features_by_layer = model({"vis": tile})

# Average the final layer's patch features into one vector.
embedding = features_by_layer[-1].mean(dim=1)
print(embedding.shape)  # torch.Size([1, 768])
```

Ideas for using embeddings:

- Find lunar tiles that look similar to a selected tile.
- Cluster a region into groups without supplying labels.
- Find unusual tiles for human inspection.
- Train a simple classifier when only a small labeled dataset is available.

## 10. Use case: multimodal generation

Lunar-FM can generate one modality while conditioning on others. For example, it can explore relationships such as DTM to slope or DTM plus observation metadata to visible imagery.

Start the notebook with:

```bash
source .venv/bin/activate
jupyter lab examples/generate_images.ipynb
```

The notebook's parameter cell asks for:

```python
data_parquet_file = "data/<parquet_file>.parquet"
data_root = "data/<data_sample_folder>"
in_domains = ["dtm", "metadata"]
out_domains = ["vis"]
device = "cpu"
```

Generation is useful for qualitative exploration and checking whether the model learned relationships between modalities. Generated values are not instrument-grade measurements and can have incorrect absolute values or coordinates.

## 11. Can it select a lunar terrain path?

Lunar-FM can contribute terrain information, but it is not a complete path planner.

A practical system would contain three parts:

```text
1. Perception
   Lunar-FM -> craters, terrain features, or learned hazard scores

2. Physical environment model
   DEM + Sun geometry -> slope, visibility, and time-dependent shadows

3. Planner
   Hazards + shadows + rover limits -> route and traversal schedule
```

A terrain cost map might combine distance, slope, crater risk, illumination, energy, and thermal constraints:

```text
cost = distance + slope penalty + hazard penalty + shadow penalty + energy penalty
```

An A*, D* Lite, or another time-dependent planner can search this cost map. For a real rover, the planner must also consider vehicle width, clearance, maximum slope, cross-slope, turn radius, speed, battery state, communication, and uncertainty.

## 12. Can it predict shadows at a specific time?

Not reliably by itself.

Lunar-FM learned from observation geometry and can generate imagery with approximate illumination patterns. However, its metadata does not contain a timestamp, and its generated shadows are qualitative.

A physically grounded shadow calculation needs:

1. A UTC date and time
2. NASA/JPL SPICE kernels
3. The Sun direction in a Moon-fixed reference frame
4. A georeferenced, high-resolution lunar DEM
5. Ray tracing or horizon testing across the DEM

For each terrain cell:

```text
if the surface faces away from the Sun:
    cell is unlit
else if terrain blocks the ray toward the Sun:
    cell is in terrain shadow
else:
    cell is illuminated
```

Repeating the calculation over time produces a three-dimensional shadow dataset:

```text
shadow[y, x, time]
```

This can answer questions such as:

- Is this location illuminated at a particular UTC time?
- When does sunlight first reach it?
- How long will it remain continuously illuminated?
- Which route minimizes travel through shadow?
- Where can a solar-powered rover stop to recharge?

Useful authoritative references are the [NASA/JPL SPICE overview](https://naif.jpl.nasa.gov/naif/spiceconcept.html), [`subslr` sub-solar point documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/subslr_c.html), and [`illumf` illumination documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/IDL/icy/cspice_illumf.html).

## 13. Fine-tuning for a new task

Fine-tuning teaches the backbone to produce a new output. Example new tasks include:

- Boulder detection
- Landing-hazard segmentation
- Geological-unit classification
- Regolith-property regression
- Permanently shadowed-region classification
- Traversability or terrain-risk prediction

LoRA is a practical starting point because it changes a relatively small number of parameters. For example:

```bash
PYTHONPATH=. terratorch fit \
  --config terratorch_integration/configs/imp/ni_lfm_ps8_lora.yaml
```

Long training runs are better suited to a CUDA GPU or compute cluster. The repository contains PBS and SLURM examples under `examples/`.

## 14. What should not be done with this model?

Do not use Lunar-FM predictions alone for:

- Landing-site certification
- Final rover safety clearance
- Exact geodetic positioning
- Instrument-grade elevation reconstruction
- Confirmation of water ice
- Mission-critical time-specific shadow predictions

For those applications, combine model predictions with physical modeling, calibrated instruments, georeferenced products, uncertainty estimates, and expert review.

## 15. Suggested beginner learning path

1. Run the IMP evaluation command and read the reported metrics.
2. Open the supplied IMP and crater prediction figures.
3. Run the feature-extraction example with a sample tile.
4. Learn how a TerraTorch YAML connects a dataset, backbone, decoder, and loss.
5. Try a one-epoch LoRA fine-tuning experiment.
6. Build a simple cost map from slope and crater detections.
7. Add a physics-based shadow layer using SPICE and a DEM.
8. Add a path planner only after validating every input layer.

## 16. Where to learn more

- [Project README](README.md)
- [TerraTorch integration guide](terratorch_integration/README.md)
- [Generation notebook](examples/generate_images.ipynb)
- [NASA-IBM Lunar-FM model page](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)
- [TerraTorch documentation](https://torchgeo.org/terratorch/quick_start/)
- [NASA/JPL SPICE](https://naif.jpl.nasa.gov/naif/)

