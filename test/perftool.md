## Intro

This README is the guidance of CATCH op's perftool.

## Usage

First, config the env variable to open the tool:
- `export TEST_WITH_PERFTOOL=ON`;

Then, execute the op's test file normally, and the corresponding filtered kernel calls report will be generated under `catch/test/torch_ops/ops_kernel_report`:
- single file: `cd catch/test/torch_ops && python test_xxx.py`;(Using pytest is ok: `cd catch/test/torch_ops && python -m pytest test_xxx.py`)
- or execute in batches: `cd catch/test && python run_test.py -v --ignore_cnnl_blacklist -i torch_ops/`;(Using pytest is ok: `cd catch/test && python run_test.py --pytest -v --ignore_cnnl_blacklist -i torch_ops/`)

At last, execute the `perftool.py` to compare with baseline reports and generate comparison results under `catch/test/torch_ops/ops_kernel_cmp_result`:
- single file: `cd catch/test && python perftool.py -v -f test_xxx`;
- all files: `cd catch/test && python perftool.py -av`.

## Environment

Baseline reports under `catch/test/torch_ops/ops_kernel_report_baseline` are generated with the following environments:
- Python3.10.8

## How to update baseline

Just generate the op's new json report and move it from `catch/test/torch_ops/ops_kernel_report` to `catch/test/torch_ops/ops_kernel_report_baseline`, overwriting original baseline file forcibly. Plz update "Logs" simultaneously.

## Logs

Baseline reports are based on the specific commit of `catch_2.1.0_develop` branch and the history logs are listed as follows:
- **20240305**: commit SHA 1e303228, op kernel's filtered policy -- Name equals to 'Memcpy HtoD'/'Memcpy DtoH' || Name startswith 'cnnl'. Currently, several files are missing reports: `test_distributions.py`(not using testinfo()).


