PYTHON ?= python3

.PHONY: all figures clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	$(PYTHON) scripts/make_summary_figures.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
