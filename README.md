# WEC2024
Repository for meterials developed for the purpose of Warsaw Econometric Challange 2024 by Gradient Descendants team.


In order to produce results for tree-based models, please move to the scripts directory and execute following code:

```
python train_trees.py --model <model_of_choice; choices: [rf, xgboost, gbm]>
```

After running the script, results and a dataframe with predictions will be saved in the _data directory.
