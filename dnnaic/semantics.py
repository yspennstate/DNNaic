"""Direction semantics for the archived three-class benchmark.

The simulations record continuous migration under ``msprime``, whose migration
matrix is defined backwards in time: an entry ``source -> dest`` moves a lineage
from ``source`` into ``dest`` looking back, which corresponds to forward-time gene
flow from ``dest`` into ``source``. The generator therefore encodes a forward
donor -> recipient edge by setting ``msprime(source=recipient, dest=donor)``.

The archived class codes and their forward-time meanings are:

    A : P1 -> P2    (msprime source=P2, dest=P1)
    B : P2 -> P3    (msprime source=P3, dest=P2)
    C : P3 -> P2    (msprime source=P2, dest=P3)
    D : no migration

This is the mapping physically present in the released arrays: for a strong-rate
class the donor population loses private alleles into the recipient (its private
allelic richness drops while the pair-private richness rises) and the recipient
gains allelic richness. ``tests/test_direction_semantics.py`` checks this against
the archived feature matrix. Populations sit on the caterpillar tree ((P1,P2),P3),
so B and C are the same sister/outgroup pair with the flow reversed -- the
donor-versus-recipient distinction that a symmetric Patterson's D cannot make.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class MigrationCase:
    """One archived class: its forward donor/recipient and the msprime encoding."""

    label: str
    forward_donor: Optional[str]
    forward_recipient: Optional[str]
    msprime_source: Optional[str]
    msprime_dest: Optional[str]

    @property
    def forward_label(self) -> str:
        if self.forward_donor is None:
            return "no migration"
        return f"{self.forward_donor}->{self.forward_recipient}"


CASES: Dict[str, MigrationCase] = {
    "A": MigrationCase("A", "P1", "P2", "P2", "P1"),
    "B": MigrationCase("B", "P2", "P3", "P3", "P2"),
    "C": MigrationCase("C", "P3", "P2", "P2", "P3"),
    "D": MigrationCase("D", None, None, None, None),
}

FORWARD_LABELS: Dict[str, str] = {k: c.forward_label for k, c in CASES.items()}

FORWARD_EDGE_TO_CLASS: Dict[Tuple[str, str], str] = {
    (c.forward_donor, c.forward_recipient): k
    for k, c in CASES.items()
    if c.forward_donor is not None
}


def forward_edge(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return the forward (donor, recipient) for an archived class code."""
    c = CASES[label]
    return c.forward_donor, c.forward_recipient


def class_for_forward_edge(donor: str, recipient: str) -> str:
    """Return the archived class code for a forward donor -> recipient edge."""
    try:
        return FORWARD_EDGE_TO_CLASS[(donor, recipient)]
    except KeyError as exc:
        raise ValueError(
            f"forward edge {donor}->{recipient} is not one of the archived classes"
        ) from exc


def msprime_pair(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return the msprime (source, dest) the generator uses for an archived class."""
    c = CASES[label]
    return c.msprime_source, c.msprime_dest
