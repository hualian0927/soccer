import copy
import unittest
from types import SimpleNamespace

from tactical_analysis.reviewed_layers import organization_candidates, reviewed_statistics, spatial_coverage
from workflows.review.build_reviewed_tactical_layers import enrich_bundle


def event(id, time, subtype="pass", team="right", category="传接带"):
    return dict(id=id, time=time, subtype=subtype, team=team, category=category,
                publishedForStatistics=True, visionStatus="visual_supported_candidate", sceneId=0)


class ReviewedLayersTest(unittest.TestCase):
    def test_cross_sequence_has_sources_not_full_possession(self):
        events = [event("a", 1), event("b", 3, "cross")]
        original = copy.deepcopy(events)
        found = organization_candidates(events)
        self.assertEqual(found[0]["sourceEventIds"], ["a", "b"])
        self.assertFalse(found[0]["completePossession"])
        self.assertEqual(events, original)

    def test_unknown_intervening_event_breaks_chain(self):
        unknown = event("u", 2)
        unknown["publishedForStatistics"] = False
        self.assertEqual(organization_candidates([event("a", 1), unknown, event("b", 3, "cross")]), [])

    def test_team_time_cut_and_rejection_gates(self):
        for change in ({"team": "left"}, {"team": None}, {"time": 30}, {"sceneId": 2}, {"sceneId": None}, {"reviewStatus": "rejected"}):
            end = {**event("b", 3, "cross"), **change}
            self.assertEqual(organization_candidates([event("a", 1), end]), [])

    def test_goalkeeper_cycle_requires_keeper_events(self):
        events = [event("a", 10, "foot_pass", category="门将事件"), event("b", 18, "foot_pass", category="门将事件")]
        self.assertEqual(organization_candidates(events)[0]["kind"], "keeper_recycle")
        events[0]["category"] = "传接带"
        self.assertEqual(organization_candidates(events), [])

    def test_stats_exclude_rejected(self):
        bad = event("b", 3, "shot")
        bad["publishedForStatistics"] = False
        self.assertEqual(reviewed_statistics([event("a", 1), bad])["subtypes"], {"pass": 1})

    def test_spatial_missing_ball_or_team_fails(self):
        people = [SimpleNamespace(track_id=i + (10 if t == "left" else 0), team=t, role="player", pitch_x=i, pitch_y=1)
                  for t in ("left", "right") for i in range(6)]
        ball = SimpleNamespace(track_id=99, team=None, role="ball", pitch_x=1, pitch_y=1)
        def context(observations):
            return SimpleNamespace(fps=30, frames=[SimpleNamespace(time_sec=10 + i / 30, observations=observations) for i in range(-30, 31)])
        self.assertTrue(spatial_coverage(context(people + [ball]), 10)["passed"])
        self.assertFalse(spatial_coverage(context(people), 10)["passed"])
        self.assertFalse(spatial_coverage(context(people[:6] + [ball]), 10)["passed"])
        ball.pitch_x = float("nan")
        self.assertFalse(spatial_coverage(context(people + [ball]), 10)["passed"])


class EnrichmentTest(unittest.TestCase):
    def setUp(self):
        events = [event("a", 1), event("b", 3, "cross")]
        for e in events:
            e.update(assistantDecision="supported", observation="画面观察", analysis="分析", limitation="边界")
        self.source = {"events": events, "layers": {"L2": {}}, "reviewMetadata": {}}
        self.annotation = {"reviewer": "internal", "scope": "抽帧", "reviews": [
            dict(id="w", level="L3", ruleKind="wide_delivery", time=0, end=4,
                 decision="supported", title="拉边传中", observation="观察", interpretation="解读", advice="建议", limitation="限制")]}
        self.evidence = {"events": [dict(id="w", time=0, title="原候选", summary="候选", evidenceStart=0, evidenceEnd=5,
                                       clipPath="clip.mp4", evidenceFrames=[{"path": "image.jpg", "time": i / 4} for i in range(21)])]}

    def test_enrichment_preserves_source_and_attaches_evidence(self):
        before = copy.deepcopy(self.source)
        result = enrich_bundle(self.source, self.annotation, self.evidence, None)
        self.assertEqual(self.source, before)
        item = result["layers"]["L3"]["items"][0]
        self.assertEqual(item["sourceEventIds"], ["a", "b"])
        self.assertEqual(len(item["inspectedFrames"]), 11)
        self.assertEqual(result["events"][0]["reviewerLabel"], "智能复核")
        self.assertEqual(result["reviewMetadata"]["api_calls"], 0)

    def test_duplicate_and_outside_window_rejected(self):
        self.annotation["reviews"].append(self.annotation["reviews"][0])
        with self.assertRaises(ValueError):
            enrich_bundle(self.source, self.annotation, self.evidence, None)
        self.annotation["reviews"].pop()
        self.annotation["reviews"][0]["end"] = 10
        with self.assertRaises(ValueError):
            enrich_bundle(self.source, self.annotation, self.evidence, None)

    def test_unsupported_chain_cannot_be_published(self):
        self.source["events"][0]["publishedForStatistics"] = False
        with self.assertRaises(ValueError):
            enrich_bundle(self.source, self.annotation, self.evidence, None)


if __name__ == "__main__":
    unittest.main()
