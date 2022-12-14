# Heavy-ball_polyak_steps

This repo contains the code generating the figures of [TO BE COMPLETED] as well as the paper itself.

It is decomposed as follow:

- the ``paper`` folder contains the source code of the arxiv version of the paper.
- in ``optimization/algorithms`` folder, each file contains an Algorithm class.
- in ``optimization/functions`` folder, each file contains a Function class.
- ``optimization/compare_methods.py`` contains a function outputing figures to compare all the algorithms of ``optimization/algorithms`` on a quadratic objective.
- ``optimization/exp_quad_long.py`` and ``optimization/exp_quad_short.py`` contains the code that generated the figures of [TO BE COMPLETED].

### Paper

All the files of the ``paper`` folder contain elements to generate the paper in pdf format.
- the ``compile.sh`` file should be run to compile the paper.
- the ``main.tex`` file contains the source code of the paper.
- the ``references.bib`` file contains the bibliography of the paper.
- the ``figures`` folder contains figures that are used in the paper and generated by the code in the ``optimization`` folder.

### List of implemented Algorithms

3 algorithms with are implemented, sometimes with several tunings, in ``optimization/algorithms``:
- Gradient Descent with constant step-size
- Gradient Descent with Polyak step-size: $\gamma = \frac{f(x)-f_\star}{\|\nabla f(x)\|^2}$
- Gradient Descent with a variant of Polyak step-size: $\gamma = \frac{2(f(x)-f_\star)}{\|\nabla f(x)\|^2}$
- Heavy-ball with constant tuning
- Heavy-ball with adaptive tuning based on Polyak step-size (Proposal of [TO BE COMPLETED]).
- Conjugate gradient

### List of implemented Functions

1 type of function is implemented in ``optimization/functions``:
- Quadratic functions that can be either randomly generated or deterministically generated.
