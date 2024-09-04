## Intro

This README is the guidance of TORCH_MLU op's perftool.

## Usage

First, config the env variable to open the tool:
- `export TEST_WITH_PERFTOOL=ON`;

Then, execute the op's test file normally, and the corresponding filtered kernel calls report will be generated under `torch_mlu/test/torch_ops/ops_kernel_report`:
- single file: `cd torch_mlu/test/torch_ops && python test_xxx.py`;(Using pytest is ok: `cd torch_mlu/test/torch_ops && python -m pytest test_xxx.py`)
- or execute in batches: `cd torch_mlu/test && python run_test.py -v --ignore_cnnl_blacklist -i torch_ops/`;(Using pytest is ok: `cd torch_mlu/test && python run_test.py --pytest -v --ignore_cnnl_blacklist -i torch_ops/`)

At last, execute the `perftool.py` to compare with baseline reports and generate comparison results under `torch_mlu/test/torch_ops/ops_kernel_cmp_result`:
- single file: `cd torch_mlu/test && python perftool.py -v -f test_xxx`;
- all files: `cd torch_mlu/test && python perftool.py -av`.

## Environment

Baseline reports under `torch_mlu/test/torch_ops/ops_kernel_report_baseline` are generated with the following environments:
- 8 cards
- Python3.10.8
- `ENABLE_FALLBACK_TO_CPU="OFF"`
- Do not install `torch_mlu_ci_overrides` IF using pytest

## How to update baseline

Just generate the op's new json report and move it from `torch_mlu/test/torch_ops/ops_kernel_report` to `torch_mlu/test/torch_ops/ops_kernel_report_baseline`, overwriting original baseline file forcibly. Plz update "Logs" simultaneously.

## Logs

Baseline reports are based on the specific commit of `r2.1_develop` branch and the history logs are listed as follows:
- **20240305**: commit SHA 1e303228, op kernel's filtered policy -- Name equals to 'Memcpy HtoD'/'Memcpy DtoH' || Name startswith 'cnnl'. Currently, several files are missing reports: `test_distributions.py`(not using testinfo()).
- **20240903**: commit SHA 7bddc741, add `device_count` info into each test file's JSON report, and update all baseline reports.


