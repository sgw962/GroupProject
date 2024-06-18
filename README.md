Group Project Code
==============================

Project containing the code for creating a model to predict next day closing stock price

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metrics tables     <- Tables of metrics results
    │   ├── stocks & trends    <- Merged data after CreateData class
    │   ├── price history      <- Use to access raw stock data
    │
    │
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── main.py    <- Use to call the features of the project
    │   │
    │   ├── data           <- Contains the CreateData class
    │   │   └── data_upload.py
    │   │
    │   │
    │   │
    │   ├── models         <- Contains the CreateModel class
    │   │   │               
    │   │   ├── create_model.py
    │   │
    │   │
    │   
    │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
