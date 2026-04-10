# Barricade Tests

Barricade currently has 129 passing tests and 9 subtests. The suite is deterministic and does not depend on network calls or flaky model output.

## Claim / Evidence / Gap

| Claim                                                        | Evidence                                                                                                                                                   | Gap                                                                                                                                                                                     |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The engine learns from prior work                            | [tests/test_phase1_persistence.py](../tests/test_phase1_persistence.py), [tests/test_extreme_validation.py](../tests/test_extreme_validation.py)           | Proven on warm-state benchmarks and repeat-task families, not on arbitrary open-world re-learning.                                                                                      |
| The engine beats an unguided baseline                        | [tests/test_barricade_runtime.py](../tests/test_barricade_runtime.py)                                                                                      | The baseline is intentionally simple and unguided; it is not a universal competitor benchmark, and the runtime now also exercises compact benchmark summaries and ablation comparisons. |
| Dispatch is safe                                             | [tests/test_phase2_dispatch.py](../tests/test_phase2_dispatch.py), [tests/test_e2e_execution.py](../tests/test_e2e_execution.py)                           | Proven in controlled repository flows, not at production scale or across every file type.                                                                                               |
| Execution sessions are real                                  | [tests/test_e2e_execution.py](../tests/test_e2e_execution.py), [tests/test_mcp_tool_descriptions.py](../tests/test_mcp_tool_descriptions.py)               | Session mechanics are proven; the suite does not claim broad code-generation quality.                                                                                                   |
| Readiness gating matters                                     | [tests/test_phase3_workflow.py](../tests/test_phase3_workflow.py), [tests/test_extreme_validation.py](../tests/test_extreme_validation.py)                 | The readiness policy is a heuristic gate, not a formal proof of task success.                                                                                                           |
| Current prompt families route to stable scaffold templates   | [tests/test_extrapolation.py](../tests/test_extrapolation.py), [tests/test_extrapolation_probe_evidence.py](../tests/test_extrapolation_probe_evidence.py) | Verified on the current controlled prompt families; this does not claim arbitrary open-world extrapolation.                                                                             |
| Novel capability synthesis works on unseen programming tasks | [tests/test_proof_probe_evidence.py](../tests/test_proof_probe_evidence.py)                                                                                | Verified on Ackermann, Sieve, Sudoku, and Collatz-style tasks under stateful thresholds; not arbitrary open-world extrapolation outside the programming-task regime.                    |

## What The Suite Proves

The tests are organized around the behaviors that matter in a governed execution engine:

1. Learning works. Warm state improves support scores, macro reuse, and repeat-task handling.
2. Baseline advantage is real. The benchmark comparison test beats an unguided baseline initialized with random DNA and no guidance stack.
3. Dispatch is safe. File updates are staged, verified, and committed only when verification passes.
4. Execution sessions are real. DNA expands into stepwise instructions and market artifacts.
5. Readiness is meaningful. `act` / `ask` / `stop` responds to task signal strength and warm state.
6. Current prompt families route to stable scaffold templates. The scaffold keeps a small set of verified DNA shapes for the current prompt families.
7. Novel capability synthesis works on unseen programming tasks. The proof probe solves Ackermann, Sieve, Sudoku, and Collatz tasks with the same scaffold family.

## Evidence Map

These are the tests that best show why Barricade is more than a token router:

| Test file                                                                                   | What it demonstrates                                                                                                         |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [tests/test_extreme_validation.py](../tests/test_extreme_validation.py)                     | Warm state improves support scores, learned macros, and repeat-task handling on held-out prompts.                            |
| [tests/test_phase1_persistence.py](../tests/test_phase1_persistence.py)                     | Learned macros, outcome memory, and persisted summaries survive across runs.                                                 |
| [tests/test_barricade_runtime.py](../tests/test_barricade_runtime.py)                       | Deterministic benchmark comparison, compact benchmark output, and ablation studies stay aligned with the runtime.            |
| [tests/test_phase2_dispatch.py](../tests/test_phase2_dispatch.py)                           | Verification failures and timeouts are blocked instead of being committed.                                                   |
| [tests/test_e2e_execution.py](../tests/test_e2e_execution.py)                               | DNA expands into a real execution session with verification and completion.                                                  |
| [tests/test_phase3_workflow.py](../tests/test_phase3_workflow.py)                           | Task classification, prior reuse, execution feedback, and readiness policy behavior.                                         |
| [tests/test_mcp_tool_descriptions.py](../tests/test_mcp_tool_descriptions.py)               | Session bridging, repair flows, completion, and learned-macro promotion.                                                     |
| [tests/test_extrapolation.py](../tests/test_extrapolation.py)                               | Structural transfer, generalization gradient, macro essence, failure modes, novel composition, and intervention sensitivity. |
| [tests/test_extrapolation_probe_evidence.py](../tests/test_extrapolation_probe_evidence.py) | Intervention sensitivity, structural transfer, generalization gradient, and macro essence evidence assertions.               |
| [tests/test_proof_probe_evidence.py](../tests/test_proof_probe_evidence.py)                 | Novel capability synthesis on unseen programming tasks, plus learning and scaffold-necessity checks.                         |

## How To Run

```bash
pytest tests -q
```

If you want a narrow smoke check while editing the product flow, start with `tests/test_phase3_workflow.py`, `tests/test_mcp_tool_descriptions.py`, `tests/test_phase2_dispatch.py`, and `tests/test_extreme_validation.py`.
