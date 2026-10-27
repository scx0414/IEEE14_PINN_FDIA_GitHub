# Third-party notices

## xLSTM core

`src/ieee14_fdia/_vendor/xlstm_core/` contains the xLSTM core used in the existing experiments. Its module declares version1.0.4. Source headers attribute copyright to NXAI GmbH and its affiliates, including the named contributors in each file. These headers are retained.

The full Apache License2.0 is retained as `src/ieee14_fdia/_vendor/XLSTM_LICENSE`. This license applies to that third-party component; the project has not selected a license for its original code.

Local modifications already used in the experiments:

1. `blocks/slstm/cell.py`: custom CUDA extension loading disabled, so the author-provided vanilla backend can run without a compiler setup.
2. `blocks/slstm/src/vanilla/__init__.py`: recurrent forward accumulation changed to list/stack form to avoid quadratic autograd CopySlices while preserving the recurrence.
3. Package imports are relative to the vendored namespace; estimator adapters reside outside the third-party core.

No MATPOWER distribution, DDET-MTD source files, private documents, or third-party paper PDFs are included. MATPOWER is an optional separately installed tool for BDD checks.
