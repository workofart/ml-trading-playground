add the episode number on each test trade graph so the animation will be more descriptive later

extract the parameters into a separate config file

when writing to the log files, also write the parameters used

(keras with tensorboard integration](https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras)

Printing out the configs:

```python
# print({k : pCfg[k] for k in pCfg})
```