from __future__ import annotations

import numpy as np
import pytest

from core.state import create_initial_state, ensure_numpy_embeddings, validate_state_shape



def test_create_initial_state_has_required_keys() -> None:
    state = create_initial_state("MKTFF")
    validate_state_shape(state)
    assert state["sequences"] == ["MKTFF"]
    assert state["history"] == []



def test_validate_state_shape_raises_for_missing_key() -> None:
    state = create_initial_state("MKTFF")
    del state["scores"]
    with pytest.raises(KeyError):
        validate_state_shape(state)



def test_ensure_numpy_embeddings_raises_for_invalid_type() -> None:
    state = create_initial_state("MKTFF")
    state["embeddings"] = [np.zeros((4,), dtype=np.float32), [1, 2, 3]]
    with pytest.raises(TypeError):
        ensure_numpy_embeddings(state)
