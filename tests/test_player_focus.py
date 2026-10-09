import unittest

from tactical_analysis.player_focus import LocalIdentities, summarize_selection


class PlayerFocusTests(unittest.TestCase):
    def test_local_ids_and_scene_reset(self):
        ids = LocalIdentities()
        self.assertEqual(ids.assign(3), ids.assign(3))
        ids.reset_scene()
        self.assertEqual(ids.assign(3), 2)

    def test_fragment_reconnection_and_collision(self):
        ids = LocalIdentities()
        box = (10,10,30,80)
        a = ids.link(1,50,box,0,30,set())
        self.assertEqual(ids.link(9,50,box,10,30,set()),a)
        self.assertNotEqual(ids.link(9,50,box,10,30,{a}),a)

    def test_far_or_edge_or_expired_anchor_not_reused(self):
        for box, gap, edge in [((800,10,820,80),1,False),((10,10,30,80),31,False),((10,10,30,80),2,True)]:
            ids = LocalIdentities()
            a = ids.link(1,50,(10,10,30,80),0,30,set(),edge)
            self.assertNotEqual(ids.link(2,50,box,gap,30,set()),a)

    def data(self):
        return {"duration":2,"fps":2,"width":100,"height":100,"frames":[
            {"frame":i,"scene":0,"players":[{"id":1,"box":[.1,.1,.1,.2],"team":"left"}],
             "balls":[[.14,.29,.02,.02]]} for i in range(4)]}

    def test_proximity_is_not_touch(self):
        report = summarize_selection(self.data(),1,0,2)
        self.assertEqual(report["coverage"],1)
        self.assertEqual(report["ball_near_feet_frames"],4)
        self.assertEqual(report["status"],"pending_visual_review")
        self.assertNotIn("touch_count",report)

    def test_missing_frames_reduce_coverage(self):
        data = self.data()
        data["frames"][1]["players"] = []
        self.assertEqual(summarize_selection(data,1,0,2)["coverage"],.75)

    def test_cut_and_invalid_range(self):
        data = self.data()
        data["frames"][2]["scene"] = 1
        with self.assertRaisesRegex(ValueError,"camera cut"):
            summarize_selection(data,1,0,2)
        for start,end in [(0,float("nan")),(-1,1),(1,0),(0,2.01)]:
            with self.assertRaises(ValueError): summarize_selection(self.data(),1,start,end)
        with self.assertRaises(ValueError): summarize_selection(self.data(),99,0,1)

    def test_jersey_color_gate(self):
        import numpy as np
        from tactical_analysis.player_tracker import shirt_class
        for color,expected in [((210,35,20),1),((245,245,245),2),((20,20,220),3),((30,110,30),0)]:
            image=np.full((100,100,3),color,dtype=np.uint8)
            self.assertEqual(shirt_class(image,(10,10,90,90)),expected)


if __name__ == "__main__":
    unittest.main()
