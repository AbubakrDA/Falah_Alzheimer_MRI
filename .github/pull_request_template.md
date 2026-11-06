## Description
Please summarize the changes and motivation for this pull request.

Fixes #(issue)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring / Performance improvement

## Medical AI & Data Governance Checklist
- [ ] **Zero Patient Leakage**: For OASIS-1 or any clinical dataset, partitioning is strictly verified at the `participant_id` level.
- [ ] **Dataset Provenance**: Datasets (Falah and OASIS-1) are never silently merged or co-mingled.
- [ ] **Imbalance Mitigation**: Class weighting or focal loss is applied where minority classes (e.g. `Moderate_Demented` < 1%) are trained.
- [ ] **Dual Runtime Safety**: Module imports safely guard against missing/failed PyTorch or TensorFlow C++ runtimes (`try ... except Exception:`).
- [ ] **Unit Tests**: All unit tests pass:
  ```bash
  python -m unittest tests/test_falah_loader.py
  python -m unittest tests/test_oasis1_integration.py
  python -m unittest tests/test_api.py
  ```
- [ ] **API & Docker**: Changes are reflected in FastAPI schemas/routes and validated in Docker.
- [ ] **Medical Disclaimer**: Scientific disclaimer is preserved in all user-facing pages and endpoints.

