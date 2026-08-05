# π0-FAST revision bridge

This directory supports a separately reported prospective cohort for the 40
previously blocked π0-FAST cells. It does **not** recover or replace the missing
historical OpenPI/RoboLab commits.

The bridge pins OpenPI `4cc827620360246dda0fa9d09a9e68269b186ecb`, whose
public config names the exact staged `openpi-assets-simeval` checkpoint, and
RoboLab `0aef241fb088ca21bb4ebd24448940ed56620d17`. Historical seeds 8300–8309,
the failed V2-A008 probe, and bridge seeds 8310–8329 remain three separate
evidence layers. Never report a pooled 30-pair π0-FAST denominator.

Run the v3 validator and the amendment's model-blind gates first. The only
initially permitted inference is the three-request fixed-observation gate:
LEFT, byte-identical repeated LEFT, and RIGHT at one sampling seed. Behavioral
execution releases only if repeat actions match exactly, LEFT/RIGHT tokenizer
bytes differ, and LEFT/RIGHT action tensors differ with nonzero RMS.
