import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tactical_analysis.models import FrameState, Observation, AnalysisContext
from tactical_analysis.formation_evidence import describe_window, frame_distribution
from tactical_analysis.analyzers.formation import estimate_frame
from tactical_analysis.analyzers.set_pieces import SetPieceDeliveryAnalyzer
from tactical_analysis.analyzers.goalkeeper import GoalkeeperInterventionAnalyzer


def formation(time=0, count=10, keeper=True):
    coords = [(-30,y) for y in (-24,-8,8,24)] + [(0,y) for y in (-20,0,20)] + [(25,y) for y in (-20,0,20)]
    obs = [Observation(int(time*30)+1,time,i,"player","left",x,y,{"x":100+i*20,"y":200,"w":12,"h":30}) for i,(x,y) in enumerate(coords[:count])]
    if keeper:obs.append(Observation(int(time*30)+1,time,99,"goalkeeper","left",-50,0))
    return FrameState(int(time*30)+1,time,obs)


class FormationEvidenceTest(unittest.TestCase):
    def test_partial_does_not_invent_players(self):
        f=formation()
        f.observations=[f.observations[i] for i in (0,1,2,4,5,7,8,10)]
        result=frame_distribution(f,"left")
        self.assertEqual(sum(result["line_counts"]),7)
        self.assertEqual(sum(map(len,result["line_members"])),7)
        self.assertFalse(result["complete_visible_team"])
        self.assertTrue(result["formation"].startswith("局部"))

    def test_full_stable_template_needs_direction_and_time(self):
        frames=[formation(i/2) for i in range(5)]
        self.assertTrue(describe_window(frames,"left",1,[])["templateSupported"])
        self.assertFalse(describe_window([formation(0)],"left",0,[])["templateSupported"])
        self.assertFalse(describe_window([formation(i/2,keeper=False) for i in range(5)],"left",1,[])["templateSupported"])
        self.assertFalse(describe_window(frames,"left",1,[.75,1.25])["templateSupported"])

    def test_invalid_boxes_and_excess_players(self):
        f=formation()
        f.observations[0]=replace(f.observations[0],bbox={"x":math.nan,"y":0,"w":10,"h":10})
        self.assertIsNone(frame_distribution(f,"left"))
        f=formation()
        f.observations.append(replace(f.observations[0],track_id=111))
        self.assertIsNone(estimate_frame(f,"left",6))

    def test_outside_coordinates_not_in_lines(self):
        f=formation()
        f.observations[0]=replace(f.observations[0],pitch_x=100)
        self.assertEqual(estimate_frame(f,"left",6)["visible_players"],9)

    def test_landing_needs_contiguous_same_player_support(self):
        def sample(t,x=10,track=1):
            f=FrameState(int(t*30)+1,t,[Observation(1,t,track,"player","left",x,0),Observation(1,t,99,"ball",None,x,0)])
            return {"frame":f,"x":x,"y":0}
        start=sample(0,0)
        call=lambda samples:SetPieceDeliveryAnalyzer._find_landing(samples,0,0,0,6,3,10)
        self.assertIsNone(call([start,sample(1)]))
        self.assertIsNone(call([start,sample(.2,60),sample(.3,60),sample(.4,60)])["landing_track_id"])
        self.assertIsNone(call([start,sample(.2)])["landing_track_id"])
        self.assertIsNone(call([start,sample(.2),sample(.3,track=2),sample(.4)])["landing_track_id"])
        self.assertEqual(call([start,sample(.2),sample(.3),sample(.4)])["landing_track_id"],1)

    def test_future_shot_does_not_turn_contact_into_save(self):
        frames=[]
        for i,distance in enumerate((12,10,8,5,2,0)):
            t=i*.2
            frames.append(FrameState(i+1,t,[Observation(i+1,t,1,"goalkeeper","right",49,0),Observation(i+1,t,99,"ball",None,49-distance,0)]))
        shot=SimpleNamespace(team="left",time_sec=1.3,actor_track_id=20,event_id="future")
        config={"minimum_proximity_samples":3,"minimum_proximity_duration_sec":.4,"possession_confirm_samples":1}
        with patch("tactical_analysis.analyzers.goalkeeper.EventTimelineAnalyzer.analyze",return_value=SimpleNamespace(events=[SimpleNamespace(**vars(shot),event_type="shot_candidate")])):
            output=GoalkeeperInterventionAnalyzer(config).analyze(AnalysisContext(Path("x"),None,5,frames))
        self.assertEqual(len(output.events),1)
        self.assertIsNone(output.events[0].metrics["shot_event_id"])
        self.assertEqual(output.findings[0].category,"goalkeeper_contact_candidate")

    def test_keeper_speed_uses_movement_interval_not_whole_episode(self):
        frames=[]
        for i in range(6):
            t=i*.2;x=40 if i==0 else 43;distance=12 if i==0 else 0 if i==1 else 1
            frames.append(FrameState(i+1,t,[Observation(i+1,t,1,"goalkeeper","right",x,0),Observation(i+1,t,99,"ball",None,x-distance,0)]))
        with patch("tactical_analysis.analyzers.goalkeeper.EventTimelineAnalyzer.analyze",return_value=SimpleNamespace(events=[])):
            output=GoalkeeperInterventionAnalyzer({"minimum_proximity_samples":3,"possession_confirm_samples":1}).analyze(AnalysisContext(Path("x"),None,5,frames))
        self.assertEqual(output.events,[])


if __name__=="__main__":unittest.main()
