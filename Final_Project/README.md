# Breast Cancer Predictor

Given features from breat tissue, the objective of this project is to predict whether the cancer is benign or malignant.

## Prerequistes

Install the necessary python packages.

```
pip install -r requirements.txt 
```
 
## Train the model 

Run `python -m model.model` from the **root** directory.

After training the model, the trained weights and the optimizers are saved in 

```bash
├── model
|    ├── results
|           ├── model.pth
|           ├── optimizer.pth
```

## Running The Flask App

Run the **app.py** file from the **root** directory

```
python app.py
```

Go to **localhost:5000** to access the application.

## References

This flask application in based on the models that were provided during the tutorial by John Wu and Tiffany Wang. 

