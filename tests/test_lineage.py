from adie.lineage.graph import LineageGraph


def test_upstream_cone_orders_by_proximity(manifest_path):
    # upstream_cone/downstream_cone return dbt's verbose unique_ids
    # ("model.adie.trips_enriched"); display_name() maps back to the bare
    # model name everywhere else in this codebase (rca/scorer.py included).
    lg = LineageGraph.from_manifest(manifest_path)
    cone = [lg.display_name(uid) for uid in lg.upstream_cone("fct_trips_daily")]
    assert cone[0] == "trips_enriched"  # the direct (1-hop) parent must come first
    assert "trips" in cone
    assert "stg_zones" in cone


def test_upstream_cone_excludes_test_nodes_by_default(manifest_path):
    lg = LineageGraph.from_manifest(manifest_path)
    cone = lg.upstream_cone("fct_trips_daily")
    for node in cone:
        assert lg.graph.nodes[node]["resource_type"] in {"model", "source", "seed"}


def test_downstream_cone_is_blast_radius(manifest_path):
    lg = LineageGraph.from_manifest(manifest_path)
    downstream = [lg.display_name(uid) for uid in lg.downstream_cone("trips")]
    assert "stg_trips" in downstream
    assert "fct_trips_daily" in downstream


def test_distance_matches_cone_order(manifest_path):
    lg = LineageGraph.from_manifest(manifest_path)
    assert lg.distance("trips_enriched", "fct_trips_daily") == 1
    assert lg.distance("trips", "fct_trips_daily") == 3


def test_unknown_asset_returns_empty_cone(manifest_path):
    lg = LineageGraph.from_manifest(manifest_path)
    assert lg.upstream_cone("does_not_exist") == []
