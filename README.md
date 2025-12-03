# y

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analyser le jeu de données pour établir la corrélation entre les habitudes de vie (alimentation et activité physique) et le niveau d'obésité des individus.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│   └── └── ObesityDataSet_raw_and_data_sinthetic.csv        <- Notre CSV
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         projet_aa and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── projet_aa   <- Source code for use in this project.
    │
    ├── src     <- Contient notre code         
    │   ├── __init__.py 
    │   ├── clustering.py       <- Code pour kmeans et CAH
    │   ├── cross_validation.py <- Code pour la crossvalidation 
    │   ├── main.py             <- Code princial / execution 
    │   ├── models_dt.py        <- Code pour crée + entrainer +recherche d'hyperparamettre du modele arbre de décision 
    │   ├── models_rf.py        <- Code pour crée + entrainer +recherche d'hyperparamettre du modele foret aléatoire 
    │   ├── preprocessing.py    <- Prétraitement des data 
    │   ├── visualisation.py    <- Création du tableau ROC

```

--------

