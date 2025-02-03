from cocoex import Suite, Observer

suite = Suite("bbob-noisy","", "dimensions: 5") # https://numbbo.github.io/coco-doc/C/#suite-parameters
problem = suite.get_problem('bbob_noisy_f101_i01_d05')
print(len(suite.ids()))