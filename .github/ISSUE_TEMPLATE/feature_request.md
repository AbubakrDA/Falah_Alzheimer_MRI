---
name: Feature Request
about: Suggest an idea or architectural improvement for NeuroVision
title: "[FEATURE]: "
labels: enhancement
assignees: ''
---

### Feature Overview
A clear and concise description of the requested feature or clinical capability.

### Clinical & Scientific Motivation
Is your feature request related to a specific neuroimaging requirement or limitation? (e.g. OASIS-2 longitudinal tracking, ADNI co-registration, 3D volumetric CNN, 3D ViT, Swin UNETR).

### Proposed Architectural Design
Describe how this feature should integrate with the existing modular architecture:
- `src/data/`: Data ingestion, parsing, or participant-level splitting
- `src/models/`: Deep learning backbone (PyTorch / Keras)
- `src/explainability/`: Attention rollout, Integrated Gradients, or Grad-CAM
- `src/api/`: FastAPI endpoints
- `pages/`: Streamlit interactive UI

### Alternatives Considered
A clear and concise description of any alternative solutions or features you've considered.

### Additional Context
Add any other context, medical citations, or mockups about the feature request here.

