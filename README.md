# Heavy-ball_polyak_steps

This repo contains the code generating the figures of [TO BE COMPLETED].

It is decomposed as follow:

- in optimization/algorithms folder, each file contains an Algorithm class.
- in optimization/functions folder, each file contains a Function class.
- optimization/compare_methods.py contains a function outputing figures to compare all the algorithms of ``optimization/algorithms`` on a quadratic objective.
- optimization/exp_quad_long.py and optimization/exp_quad_short.py contains the code that generated the figures of [TO BE COMPLETED].

### List of implemented Algorithms

3 algorithms with are implemented, sometimes with several tunings, in optimization/algorithms:
- Gradient Descent with constant step-size
- Gradient Descent with Polyak step-size: $\gamma = \frac{f(x)-f_\star}{\|\nabla f(x)\|^2}$
- Gradient Descent with a variant of Polyak step-size: $\gamma = \frac{2(f(x)-f_\star)}{\|\nabla f(x)\|^2}$
- Heavy-ball with constant tuning
- Heavy-ball with adaptive tuning based on Polyak step-size (Proposal of [TO BE COMPLETED]).
- Conjugate gradient

### List of implemented Functions

1 type of function is implemented in optimization/functions:
- Quadratic functions
