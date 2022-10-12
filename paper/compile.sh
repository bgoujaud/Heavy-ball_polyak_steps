#!/usr/bin/env bash

echo Runing short experiment
python ../optimization/exp_quad_short.py

echo Runing long experiment
python ../optimization/exp_quad_long.py

echo Compiling paper
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex

rm *.aux *.log *.toc *.bbl *.blg *.out
