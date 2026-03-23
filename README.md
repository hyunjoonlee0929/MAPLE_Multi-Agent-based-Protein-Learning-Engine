# MAPLE (Multi-Agent Protein Learning Engine)

MAPLE is a multi-agent system that autonomously explores protein sequence space and optimizes enzyme function through iterative learning.

## MVP Scope
- Random mutation-based sequence exploration
- Dummy structure prediction interface (future AlphaFold2/ESMFold compatible)
- Random deterministic embeddings + simple PyTorch MLP for property prediction
- Evaluation, ranking, top-k selection, and iterative optimization loop

## Run
```bash
cd /Users/hyunjoon/codex/MAPLE
python3 main.py
```

## Agents
- planner_agent: runtime config/state validation
- sequence_agent: mutation/generation
- structure_agent: placeholder structure outputs
- property_agent: embeddings + stability/activity prediction
- optimization_agent: top-k elite evolution
- evaluation_agent: filtering, scoring, ranking
