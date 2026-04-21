# Methodology

## Development Approach

The project follows a staged prototype approach:

### Stage 1: Pipeline Demonstration
- establish webcam input
- preprocess live frames
- create real-time overlay and alarm behavior
- validate end-to-end responsiveness

### Stage 2: Early Detection Logic
- implement heuristic suspicious-pattern screening
- verify threshold tuning under monitor-capture conditions
- test live performance and frame stability

### Stage 3: CNN Integration
- collect labeled X-ray data
- train a CNN through transfer learning
- export and integrate the trained model
- compare CNN performance against baseline heuristic screening

## Evaluation Criteria
- functional live frame acquisition
- detection responsiveness
- user-visible alert behavior
- modular readiness for trained model integration
