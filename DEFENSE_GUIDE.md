The complete standalone defense guide is provided separately as SECOM_Defense_Study_Guide.md. A concise quick reference is below.

- Decision: flag production entity for more quality inspection (not scrap).
- Target label: -1 Pass -> 0; +1 Fail -> 1.
- 1567 records, 590 sensor columns, 104 fails; 70/30 stratified seed 42.
- Metric: failure recall = TP/(TP+FN); baseline always Pass, 0% failure recall.
- Test n=471, actual fails=31; Logistic TP9, FN22, FP65; Tree TP11, FN20, FP69.
- Worst exploratory slice: Logistic, >30 missing sensors, sample n220, failures13, missed12.
- Business threshold: tree preferable to baseline iff cost of a missed failure / cost of extra inspection > 69/11 (~6.27); actual costs unknown.
- Recommendation: human-review research prototype only; no automated scrap/shipment.
- Unverified: training-only tuning and PCA numerical conclusions require running raw data.
