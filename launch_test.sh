#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
pytest --cov-config=.coveragerc --cov-report term-missing --cov=. --junitxml=/tmp/xunit.xml dreem_learning_open/test/test_models_features_only.py
