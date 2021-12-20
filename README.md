# views_competition

> Code and data used in the clean-up and evaluation of the 2020 ViEWS prediction competition.

## Installation
Install the requirements in a fresh virtual environment. To make one, navigate to this project's directory and run:
```zsh
python -m venv ./.venv
```
Activate the virtual environment with:
```zsh
source .venv/bin/activate
```
Finally, install the requirements into the environment with:
```zsh
pip install --editable .
pip install -r requirements.txt
```

## Usage
On the command line, run `python views_competition/run.py` with the root of this project as your current working directory. 

The script will first clean up and prepare the raw participant submissions, and then proceeds with their evaluation. Figure and table outputs are written to a `output` directory in the root of this project. Logs are written to the `logs` directory.

Parts of the script can be skipped or individually run by adjusting `views_competition/config.py`.

The total running time is approximately five hours.
