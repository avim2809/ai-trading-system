"""Strategy-family lifecycle (ticket P5-01): state machine, G-PAPER evaluator, decommission combiner, gated sleeve.

"Sleeve" elsewhere in the repo means a capital bucket; lifecycle states apply to strategy FAMILIES. Nothing here is wired into
the live allocator, ``SLEEVE_REGISTRY`` or the engine (P6-01). Thresholds come from ``config/gates.yaml`` (never hard-coded).
"""
