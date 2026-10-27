# Detection delay comparison

Full NAB corpus: 58 files and 116 labelled anomaly windows. Delay is measured from the start of a labelled window to the first predicted event start inside it. Medians and means use **detected windows only**; a missed window has no delay value. The window-position percentage makes files with different sampling rates and window widths easier to compare. Lower delay can still come with lower recall, so read it beside the number of windows found.

Published models use NAB's stored per-point scores at its standard-profile threshold, with each new threshold crossing counted as an event. Sentinel uses the latest saved full-corpus run tagged `engine default (final)`. These event rules differ between systems. NAB windows usually centre the marked anomaly, which is why most medians are near 50% of the window width.

| Model | Event-F1 | Windows found / 116 | Median delay (readings) | Median position | Mean position |
|---|---:|---:|---:|---:|---:|
| Sentinel current default | 0.691 | 76 | 100.0 | 49.7% | 50.6% |
| CAD OSE | 0.676 | 92 | 111.0 | 49.7% | 48.9% |
| Numenta HTM | 0.527 | 96 | 100.5 | 49.6% | 45.7% |
| ARTime | 0.483 | 106 | 105.0 | 49.9% | 47.3% |
| Numenta TM | 0.457 | 91 | 102.0 | 49.4% | 44.7% |
| Relative Entropy | 0.452 | 78 | 116.5 | 49.8% | 48.9% |
| HTM Java | 0.444 | 93 | 101.0 | 49.1% | 43.8% |
| earthgecko Skyline | 0.372 | 87 | 115.0 | 48.8% | 44.0% |
| KNN CAD | 0.350 | 91 | 100.0 | 48.8% | 44.4% |
| Twitter ADVec | 0.341 | 77 | 137.0 | 49.7% | 48.7% |
| Random Cut Forest | 0.333 | 87 | 102.0 | 49.4% | 46.9% |
| Etsy Skyline | 0.295 | 72 | 110.5 | 48.3% | 42.9% |
| Windowed Gaussian (NAB) | 0.283 | 73 | 100.0 | 48.8% | 44.9% |
| EXPoSE | 0.161 | 51 | 112.0 | 47.4% | 45.9% |
| Random (NAB) | 0.160 | 51 | 141.0 | 49.7% | 48.5% |
| Null (NAB) | 0.000 | 0 | — | — | — |
