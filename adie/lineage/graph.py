"""
Lineage plane: turns the dbt manifest into a queryable NetworkX graph.

Deliberately does NOT run causal discovery. dbt's manifest already encodes
the true dependency DAG (every `ref()`/`source()` call), so this module's
only job is to load that structure and answer two questions cheaply:

  - upstream_cone(asset)   -> every ancestor that could be a root cause
  - downstream_cone(asset) -> everything that would be affected (blast radius)
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


class LineageGraph:
    def __init__(self, graph: nx.DiGraph):
        self._g = graph

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "LineageGraph":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())

        g = nx.DiGraph()

        nodes: dict[str, dict] = {}
        nodes.update(manifest.get("nodes", {}))
        nodes.update(manifest.get("sources", {}))

        for unique_id, node in nodes.items():
            g.add_node(
                unique_id,
                name=node.get("name"),
                resource_type=node.get("resource_type"),
                # dbt manifest columns are keyed by a dict even when empty
                columns=list(node.get("columns", {}).keys()),
            )

        for unique_id, node in manifest.get("nodes", {}).items():
            for dep in node.get("depends_on", {}).get("nodes", []):
                if dep in g:
                    g.add_edge(dep, unique_id)

        return cls(g)

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    def node_for_name(self, name: str) -> str | None:
        """dbt unique_ids are verbose (model.adie.fct_trips_daily); allow lookup by bare name."""
        if name in self._g:
            return name
        matches = [n for n, d in self._g.nodes(data=True) if d.get("name") == name]
        return matches[0] if matches else None

    def upstream_cone(self, asset: str, kinds: set[str] | None = {"model", "source", "seed"}) -> list[str]:
        """Every ancestor of `asset`, nearest first — the RCA search space.

        Restricted to real data assets (model/source/seed) by default so
        dbt's generic test nodes -- which also sit in the dependency graph --
        never show up as root-cause candidates; pass kinds=None for the raw
        ancestor set including tests."""
        uid = self.node_for_name(asset)
        if uid is None:
            return []
        ancestors = nx.ancestors(self._g, uid)
        if kinds is not None:
            ancestors = {
                a for a in ancestors if self._g.nodes[a].get("resource_type") in kinds
            }
        depth = {
            a: min(
                len(p) - 1
                for p in nx.all_simple_paths(self._g, a, uid)
            )
            if nx.has_path(self._g, a, uid)
            else 10_000
            for a in ancestors
        }
        return sorted(ancestors, key=lambda a: depth[a])

    def downstream_cone(self, asset: str, kinds: set[str] | None = {"model", "source", "seed"}) -> list[str]:
        """Blast radius: everything that depends on `asset`, transitively."""
        uid = self.node_for_name(asset)
        if uid is None:
            return []
        descendants = nx.descendants(self._g, uid)
        if kinds is not None:
            descendants = {
                d for d in descendants if self._g.nodes[d].get("resource_type") in kinds
            }
        return sorted(descendants)

    def distance(self, from_asset: str, to_asset: str) -> int | None:
        src, dst = self.node_for_name(from_asset), self.node_for_name(to_asset)
        if src is None or dst is None or not nx.has_path(self._g, src, dst):
            return None
        return nx.shortest_path_length(self._g, src, dst)

    def display_name(self, unique_id: str) -> str:
        return self._g.nodes[unique_id].get("name", unique_id)
