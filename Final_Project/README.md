# Breast Cancer Predictor

Given features from breat tissue, the objective of this project is to predict whether the cancer is benign or malignant.

## Prerequistes

Install the necessary python packages.

```
pip install -r requirements.txt 
```
 
## Train the model 

Run `cd model` followed by `python model.py` from the **root** directory.

After training the model, the hyperparameters are saved in 

```bash
├── model
|    model.joblib
```

## Running The Flask App

Run the **app.py** file from the **root** directory.

To go back to the **root** directory, write `cd ..`
```
python app.py
```

Go to **localhost:5000** to access the application.

## References

This flask application in based on the models that were provided during the tutorial by John Wu and Tiffany Wang. 

