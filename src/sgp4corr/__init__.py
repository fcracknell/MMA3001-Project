"""SGP4 orbit-prediction error correction with gradient boosting.

Pipeline modules, run in order from the repository root:

1. `sgp4corr.download`: select LEO debris objects, download element sets and space weather.
2. `sgp4corr.pairs`: propagate element sets with SGP4 and compute residuals
    (uses `sgp4corr.frames` for the RTN frame).
3. `sgp4corr.features`: build model inputs and flag outliers.
4. `sgp4corr.models`: split data and train ridge and gradient-boosting corrections;
    `sgp4corr.evaluate` computes test metrics and figures.
5. `sgp4corr.performance`: runtime, profiling and feature importance.

See the README for setup and reproduction steps.
"""