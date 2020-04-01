#!/usr/bin/env bash

mkdir temp
cp *.py temp
(
    cd temp
    # normal pandas installation with pip won't work, see e.g. https://medium.com/@korniichuk/lambda-with-pandas-fd81aa2ff25e
    # alternatively, layers could (and should) be used
    wget https://files.pythonhosted.org/packages/4a/6a/94b219b8ea0f2d580169e85ed1edc0163743f55aaeca8a44c2e8fc1e344e/pandas-1.0.3-cp37-cp37m-manylinux1_x86_64.whl
    wget https://files.pythonhosted.org/packages/b7/ce/d0b92f0283faa4da76ea82587ff9da70104e81f59ba14f76c87e4196254e/numpy-1.18.2-cp37-cp37m-manylinux1_x86_64.whl
    unzip pandas-1.0.3-cp37-cp37m-manylinux1_x86_64.whl
    unzip numpy-1.18.2-cp37-cp37m-manylinux1_x86_64.whl
    pip install -r ../requirements.txt -t .

    rm ../cleaningLambda.zip
    zip -r ../cleaningLambda.zip . -x "*.zip" "*.md" "*.sh" "*.whl" "__pycache__" "*.dist-info"
)
rm -r temp
