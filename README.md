# IDEA 2.0 – Intelligent Data Exchange Alliance

**Repository:** `idea-open-source`  
This repository contains Python code for generating profiles from Floating Car Data (FCD),
as well as validating roadwork data using FCD during roadwork periods.

If you want to make changes or add functionality you can do this by opening a pull request.

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

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environments.

### Step 1: Install uv

Install uv using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Set up the project

Clone the repository and install the dependencies:

```bash
git clone idea-open-source.git
cd idea-open-source
```

Install the environment (including dev dependencies):

```bash
uv sync --dev
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
uv run pytest
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
