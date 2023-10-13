# An Agent-Based Model of The Emergency Medical Services System in Nelson Mandela Bay
This GitHub repository contains the scripts used to run the base model and conduct the scenario and sensitivity analyses. It does not contain any input data.

- **model_functions.py** contains all the functions used by the Agent-Based Model, including *run_sim()*, which runs the model itself.
- **scenarios.py** runs the model for the base scenario, as well as for additional scenarios, with parameter values specified in a [Google Sheet](https://docs.google.com/spreadsheets/d/1Urk8KPvZEouzupBZm7YBi60_bBvgJy3O5jjzCZfnMZE/edit?usp=sharing).
- **sensitivity.py** runs the model using parameters generated by Latin Hypercube Sampling.
- **run_scenarios.sh** and **run_sensitivity.sh** are bash scripts that run *scenarios.py* and *sensitivity.py* respectively on 50 cores in parallel.
   
