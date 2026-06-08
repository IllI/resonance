# CHRONOS Time-Emission Summary

- backend: `tpu`
- devices: `8`
- D: `512`
- sparse_block: `32`
- layers: `6`
- blocks: `20`
- block_size: `100`

| condition | delta_mean_ns | MW_p | KS_p | mean_acc | sideband_acc | combined_acc | shuffled_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active | -1111 | 0.06607 | 0.08658 | 1.000 | 0.500 | 0.500 | 1.000 |
| severed | +492 | 0.3389 | 0.4266 | 0.250 | 0.500 | 0.250 | 0.250 |
| shuffled | -294 | 0.4949 | 0.5602 | 0.750 | 0.500 | 0.250 | 0.750 |
| equal_flop | +1193 | 0.4051 | 0.635 | 0.500 | 0.500 | 0.500 | 0.500 |

Interpretation gates:

- Physical coupling: active significant while severed and shuffled are null.
- Beyond mean latency: sideband or combined classifier improves over mean latency.
- Topology specificity: equal-FLOP topology control separates with matched or near-matched mean latency.
