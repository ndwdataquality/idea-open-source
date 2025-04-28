# IDEA 2.0 – Intelligent Data Exchange Alliance

**Repository:** `idea-open-source`  
This repository contains Python code for generating profiles from Floating Car Data (FCD),
as well as validating roadwork data using FCD during roadwork periods.


> **Note:** This is a _read-only mirror_ of the original repository.  
> Pull requests submitted via GitHub will not be reviewed or merged.

All changes are managed in an external source and automatically synchronized to this repository.  

---

## 📁 Project Structure

```

idea/
├── profile/       
│   ├── profile.py        # Contains the main profile generation function
│   └── util.py           # Contains the FCD-based profile generation logic
├── validation/           # Contains the roadwork validation algorithms
│   ├── validation.py     # Contains the main validation roadwork function
│   └── util.py           # Contains the validation algorithm logic
├── tests/                # Contains the unit tests
examples/
├──  calculate_minutes_no_coverage.py # Example for calculating the minutes without coverage
├──  calculate_profile.py # Example for creating a profile
├──  validate_roadwork.py # Example for validating a single segment roadwork
├──  visualization_example_cases.ipynb # Graphs of 3 example cases
```

---

## ⚙️ Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environments.

### Step 1: Install Poetry with pipx

Install poetry with pipx .  If you do not have pipx install using instructions from [pipx](https://pipxproject.github.io/pipx/installation/)

> You might need to restart terminal/IDE after installation ( in windows that is needed to be able to find the executable)

> Alternative way of installing see here [poetry](https://python-poetry.org/docs/), but pipx is recommended so we do it in the same way in the team, in the pipeline it is installed using pipx as well and in dockerfiles.

```bash
pipx install poetry==2.1.2
```

> use same version in pipeline, setting no version should install latest version.

Configure poetry to create a virtual environment in the project folder under .venv, otherwise it will be installed in the user folder with the project name .

```bash
poetry config virtualenvs.in-project true  
```

### Step 2: Set up the project

Clone the repository and install the dependencies:

```bash
git clone idea-open-source.git
cd idea-open-source
```
Install the Poetry environment:
```bash
poetry install
```
---
## 🧠 Functionality

### Profile Generation

The core function `calculate_profile` resides in `idea/profile/`. It takes Floating Car Data as input and returns a profile object that can be used for downstream validation.

#### Example

```python
from idea.profile.profile import calculate_profile

profile = calculate_profile(fcd_data)
```

### Roadwork Validation

The core function `validate_roadwork` resides in `idea/validation/`. It takes the Profile and the Floating Car data during the roadwork as input and calculates a status by minute.

---

## 🧪 Testing

Running tests

```bash
poetry run pytest
```

---

## FCD Coverage Values

Explanation of the `fcd` column:

| Value   | Meaning                               |
|---------|---------------------------------------|
| `null`  | No data (missing)                     |
| `0`     | No vehicles in this minute            |
| `1`     | One vehicle or low number of vehicles |
| `2-10`  | Proportional scale (e.g. 10 = 100%)   |
---
