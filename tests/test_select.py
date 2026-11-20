from academicdigest.models import Cluster
from academicdigest.pipeline.select import already_sent, select

from .conftest import make_item

TITLES = {"ml": "ML", "science": "Science"}


def cluster(title, category, score, **kw):
    c = Cluster(items=[make_item(title, **kw)], category=category, score=score)
    return c


def test_max_total_trims_weakest_category_picks_not_quiet_sections():
    clusters = [cluster(f"ml paper {i}", "ml", 20 - i) for i in range(10)]
    clusters.append(cluster("nature paper", "science", 4.5))
    cfg = {"selection": {"top_n": 3, "per_category": 5, "min_score": 4.0, "max_total": 6}}
    d = select(clusters, cfg, TITLES, "2026-11-20")
    sections = {s.key: [c.title for c in s.clusters] for s in d.sections}
    assert sections["top"] == ["ml paper 0", "ml paper 1", "ml paper 2"]
    assert sections["science"] == ["nature paper"]  # kept although lowest overall
    assert d.count == 6


def test_already_sent_by_arxiv_id():
    c = cluster("Different title on HN", "ml", 9, arxiv_id="2611.00001")
    assert already_sent(c, {"arxiv:2611.00001"}, [])
    assert not already_sent(c, {"arxiv:2611.99999"}, [])
    assert already_sent(cluster("Exactly this paper title here", "ml", 9), set(), ["Exactly this paper title here"])
