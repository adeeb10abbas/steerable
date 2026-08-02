# Efficient-WAM hierarchical steering gate

Across 42 closed-loop episodes, intervention-prefix integrity passed in 6/6 groups and the switch occurred at the same action in 6/6 groups.

| Condition | Strict success | Requested side reached | Favorable shift vs native control | Median actions |
| --- | ---: | ---: | ---: | ---: |
| Static native task | 6/6 | 6/6 | - | 76 |
| Static counterfactual task | 2/6 | 5/6 | - | 400 |
| Switch: full task | 3/6 | 3/6 | 4/6 | 229 |
| Switch: subtask | 2/6 | 3/6 | 3/6 | 400 |
| Switch: atomic motion | 0/6 | 1/6 | 3/6 | 400 |
| Switch: combined + release | 1/6 | 4/6 | 5/6 | 400 |
| Switch: native-direction control | 6/6 | 6/6 | - | 53 |

## Native-left task

| Condition | Strict success | Requested side reached | Favorable shift vs native control |
| --- | ---: | ---: | ---: |
| Static native task | 3/3 | 3/3 | - |
| Static counterfactual task | 0/3 | 2/3 | - |
| Switch: full task | 0/3 | 0/3 | 1/3 |
| Switch: subtask | 0/3 | 0/3 | 0/3 |
| Switch: atomic motion | 0/3 | 0/3 | 0/3 |
| Switch: combined + release | 0/3 | 1/3 | 2/3 |
| Switch: native-direction control | 3/3 | 3/3 | - |

## Native-right task

| Condition | Strict success | Requested side reached | Favorable shift vs native control |
| --- | ---: | ---: | ---: |
| Static native task | 3/3 | 3/3 | - |
| Static counterfactual task | 2/3 | 3/3 | - |
| Switch: full task | 3/3 | 3/3 | 3/3 |
| Switch: subtask | 2/3 | 3/3 | 3/3 |
| Switch: atomic motion | 0/3 | 1/3 | 3/3 |
| Switch: combined + release | 1/3 | 3/3 | 3/3 |
| Switch: native-direction control | 3/3 | 3/3 | - |

Strict success uses the official relation geometry plus both grippers open. A favorable shift only asks whether a counterfactual command moved the final x endpoint away from the matched same-prefix native-direction control.

The released checkpoint was not trained on the paper's synthetic command-style mixture. Command-style differences here measure its zero-shot interface bandwidth, not a reproduction of Steerable Policies training.
