from academicdigest.models import MENTION
from academicdigest.pipeline.dedupe import cluster_items

from .conftest import make_item


def test_arxiv_hf_and_hn_versions_merge():
    items = [
        make_item("Sparse Mixture of Experts at Scale", "https://arxiv.org/abs/2611.00001",
                  source="arXiv", tier=3, arxiv_id="2611.00001"),
        make_item("Sparse Mixture of Experts at Scale", "https://arxiv.org/abs/2611.00001",
                  source="HF Papers", arxiv_id="2611.00001"),
        make_item("MoE at scale [pdf]", "https://arxiv.org/pdf/2611.00001v2",
                  source="Hacker News", kind=MENTION, arxiv_id="2611.00001"),
    ]
    clusters = cluster_items(items)
    assert len(clusters) == 1
    assert clusters[0].lead.source == "HF Papers"  # paper record with best tier, not the HN title


def test_journal_version_merges_by_doi():
    items = [
        make_item("Preprint title", source="arXiv", arxiv_id="2611.00002", doi="10.1038/s41586-026-00001-x"),
        make_item("Published title differs", "https://www.nature.com/articles/s41586-026-00001-x",
                  source="Nature", tier=1, doi="10.1038/s41586-026-00001-x"),
    ]
    assert len(cluster_items(items)) == 1


def test_journal_version_merges_by_title():
    items = [
        make_item("Protein structure prediction with diffusion language models", source="arXiv",
                  arxiv_id="2611.00003"),
        make_item("Protein structure prediction with diffusion language models.", source="Nature", tier=1,
                  doi="10.1038/s41586-026-00002-x"),
    ]
    assert len(cluster_items(items)) == 1


def test_different_arxiv_ids_never_merge_by_title():
    items = [
        make_item("Scaling laws for sparse mixture of experts models", "https://arxiv.org/abs/2611.00004",
                  source="arXiv", arxiv_id="2611.00004"),
        make_item("Scaling laws for sparse mixture of experts models", "https://arxiv.org/abs/2611.00005",
                  source="HF Papers", arxiv_id="2611.00005"),
    ]
    assert len(cluster_items(items)) == 2


def test_same_source_never_merges_by_title():
    items = [
        make_item("Efficient attention for very long context windows", "https://nature.com/a",
                  source="Nature", doi="10.1/a"),
        make_item("Efficient attention for very long context windows", "https://nature.com/b",
                  source="Nature", doi="10.1/b"),
    ]
    assert len(cluster_items(items)) == 2
