# Q+D versus I+Q

## Workflow cost

| System | Three job times (s) | Mean | Per epoch | Saved trainable tensors | Image handling |
|---|---:|---:|---:|---:|---|
| Q+D | 217, 183, 174 | 191.3s | 31.9s | 14,688,769 | 401 images textualized once, then cached |
| I+Q | 1637, 1586, 1583 | 1602.0s | 200.2s | 24,719,873 | 4,785 image presentations per final run |

The complete I+Q job is 8.37x the Q+D job; after
normalizing by the different epoch counts, it is 6.28x
per epoch.  Timings cover model loading, final fitting, test inference, and
artifact saving on the same host and adjacent GPU slots.  These timings do not
separate training from inference, do not record peak memory, and exclude the
one-time description-generation run, so they should not be read as a
component-level cost breakdown.

## Overall item-level comparison

| n | I+Q wins | Q+D wins | Q+D RMSE | I+Q RMSE | Delta RMSE (I+Q - Q+D) | 95% paired bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|
| 145 | 74 | 71 | 0.4975 | 0.4907 | -0.0068 | [-0.0522, +0.0372] |

An item-level win means lower absolute error for that item.  The win count
measures frequency, while RMSE also reflects the magnitude of a smaller number
of large wins or losses.  The route preference is unanimous across all three
matched seeds for 37 I+Q items and
40 Q+D items; the remaining
68 items change winner across seeds.
See the CSV outputs for taxonomy, figure status, difficulty quintile,
description-length, seed stability, and the largest wins for each interface.
