# Setup exclusion: first Cosmos canonical launch

- UTC launch: 2026-08-02T15:18:27Z
- container: `v1-cosmos-canonical`
- policy episodes produced: **0**
- output files under `output/v1_cosmos_canonical`: **0**
- model request observed by policy server: **none**

Isaac Sim stopped during RTX startup before it created the RoboLab experiment
directory. Its log reported:

```text
The currently installed NVIDIA graphics driver is unsupported or has known issues.
Installed driver: 535.53
The unsupported driver range: [0.0, 535.129)
rtx driver verification failed.
```

This host has the previously validated 535-series Vulkan minor-version
reporting issue documented in the retrospective Cosmos pilot. The launch was
terminated after confirming that no environment, model request, or episode
output existed. The exact frozen evaluation was restarted with only the
targeted Isaac Kit setting
`--/rtx/verifyDriverVersion/enabled=false`; prompts, seeds, model, task,
renderer, action horizon, and metrics were unchanged.

This is a setup-invalid exclusion under the preregistered rule, not a model
failure and not part of any denominator.
