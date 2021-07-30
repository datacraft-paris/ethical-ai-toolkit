# Welcome to project Ethical AI Toolkit
![](assets/banner.png)

- **Project name**: Ethical AI Toolkit
- **Library name**: ekit
- **Authors**: Ekimetrics
- **Description**: Open source Ethical AI toolkit

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)



## Project Structure
```
- ekit/        ----- Your python library (only .py files)
- data/
    - raw/
    - processed/
- docs/                             # Documentation folder and website (.md, .ipynb) using Mkdocs
- notebooks/                        # Jupyter notebooks only (.ipynb)
- scripts/                          # Every automation script (.bat, .py, .sh)
- tests/                            # Unitary testing using pytst
- .gitignore                        # Configuration file to ignore files on Bitbucket/Github
- bitbucket-pipelines.yml           # Automation "as-code" in Bitbucket
- mkdocs.yml                        # Documentation configuration
- requirements.txt                  # Dependencies to use the library in a blank environment
- setup.py                          # Configuration file to export and package the library                   
```


## Starter package
This project has been created using the Ekimetrics Python Starter Package to enforce best coding practices, reusability and industrialization. <br>
If you have any questions please reach out to the inno team and [Théo Alves Da Costa](mailto:theo.alvesdacosta@ekimetrics.com)



## Important Commands

* `mkdocs serve` - to launch the documentation.