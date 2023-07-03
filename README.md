# Dekki
An ML based spaced repetition algorithm to help you learn faster and remember longer.

## Pre-processing the data
Download .apkg files and place them into `data/downloads`. Next, run the following command to preprocess the data.

```commandline
python build_data.py
```

## Training the model
To run training with default parameters, run the following command:

```commandline
python train.py
```

You can configure training by editing `configs.yaml`, or by passing in command line arguments. 
For example, to change the learning rate, run the following command:

```commandline
python train.py train.lr=0.001
```


## Contribute
Help us improve Dekki by contributing data, code, or ideas. You can contact me at marawan.gamal[AT]mila.quebec

We are collecting data from Anki users to improve the model @ [Dekki Data Collection](https://drive.google.com/drive/folders/18EWZD_kRBQvFpHthvZyejwAXyAnkjzBf?usp=drive_link).
