# tybalt-with-candle

This repository shows how to run a grid search hyperparameter optimization of [Tybalt](https://hpc.nih.gov/apps/Tybalt.html) using [CANDLE](https://hpc.nih.gov/apps/candle).

The repository starts tracking from the exact files located in `/usr/local/apps/Tybalt/20210709/src` as of about 11:50 AM on 8/23/21 and shows step-by-step (see the [commit log](https://github.com/andrew-weisman/tybalt-with-candle/commits/main)) how to run them using CANDLE.

These steps are inferred directly from the [guide to CANDLE on Biowulf](https://hpc.nih.gov/apps/candle); see [here](https://hpc.nih.gov/apps/candle/#usage_summary) in particular for a summary of the steps.

In summary, here are the steps required to run HPO on Tybalt using CANDLE (this is from the [commit log](https://github.com/andrew-weisman/tybalt-with-candle/commits/main)):

1. [Remove requirement](https://github.com/andrew-weisman/tybalt-with-candle/commit/721b0f18a8636ad496091a3db7abd41b5a4e6ed8) of having the type of model as a command line argument (it will simply be a hyperparameter) since we currently need to be able to run the model script like `python MY_MODEL_SCRIPT.py` with no arguments. Files affected: `options.py` and `train.py`
1. [Define the hyperparameters](https://github.com/andrew-weisman/tybalt-with-candle/commit/801bfa9a5071b29da9f12d1854590e6fdc646080) in the model script `train.py`. File affected: `train.py`
1. [Define the performance metric](https://github.com/andrew-weisman/tybalt-with-candle/commit/84fbddd2105c0ff591de80016ec0616d8ae53e91) in this case by assigning the output of the Keras `fit()` method to a variable named `history`. File affected: `train.py`
1. [Perform the copy step](https://github.com/andrew-weisman/tybalt-with-candle/commit/84be9e9a9d42c3cda2e4fd1e39b7de72ec94e46c) from the Tybalt Biowulf help page in the model script itself since each hyperparameter set will generate its own different checkpoints. File affected: `train.py`
1. [Add a sample CANDLE input file](https://github.com/andrew-weisman/tybalt-with-candle/commit/2ad3086eeec6975fe3a74107d9d9c4ecde1507d2); this file can be put in any directory, optionally an empty one, and be run using `candle submit-job candle_grid_search_for_tybalt.in`. File affected: `candle_grid_search_for_tybalt.in`
1. [Add a CSV file](https://github.com/andrew-weisman/tybalt-with-candle/commit/903d92b8f0f8cfc11fe2f9ef7dea1ca7ceebb77b) of the full grid HPO results, which was generated using `candle aggregate-results $(pwd)/last-candle-job`. This job was run using 12 GPUs (`nworkers=12`) and completed in 13 minutes. See `last-candle-job/run/*/subprocess_out_and_err.txt` for the output from each hyperparameter set. File affected: `candle_results.csv`

**Notes for Gennady:**

* As seen in the steps above, it is very easy to use CANDLE. It is not just for advanced users and can certainly be (and has been) used by beginners!
* You will not run into any issues if you follow the steps above, which come straight from the [CANDLE documentation](https://hpc.nih.gov/apps/candle).
* In particular, there are no issues or conflicts at all with the Python libraries or versions. E.g., your code does not need to depend on `python/3.7` in any way, nor do you need to know the Python packages on which CANDLE depends.
* I think you could write a simple wrapper around this in order to present it clearly to your class as you previously tried to do. I would be happy to help with this or with anything else!

**Gennady, please let me know if you have any particular questions or if something is still unclear!**
