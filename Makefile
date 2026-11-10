.PHONY: test cartpole pendulum followup-report

test:
	cd implementations/eq16 && python -m pytest -q tests
	cd implementations/eq16_followup && python -m pytest -q tests
	python -m pytest -q tests
cartpole:
	cd implementations/eq16 && python run_benchmark.py --task cartpole --smoke --device cpu
pendulum:
	cd implementations/eq16 && python run_benchmark.py --task pendulum --smoke --device cpu
followup-report:
	cd implementations/eq16_followup && python followup.py report --results-root example_results --output results/followup_analysis
