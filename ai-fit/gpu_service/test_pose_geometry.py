import copy,unittest
from types import SimpleNamespace
from pose_geometry import hand_protection,pose_difference,foot_protection,neck_protection,body_width_difference,align_upper_shoulders,torso_anchors
import numpy as np

class PoseGeometryTests(unittest.TestCase):
    def test_cropped_alignment_uses_fixed_frame_boundary_without_invented_hip_anchor(self):
        from PIL import Image
        p=self.points();p[0].y=.1;p[9].y=p[10].y=.1
        for i in (23,24):p[i].y=1.2;p[i].visibility=p[i].presence=.01
        after=copy.deepcopy(p);after[11].y=after[12].y=.238
        pixels=np.broadcast_to((np.arange(400)%256).astype(np.uint8)[:,None,None],(400,300,3)).copy()
        image=Image.fromarray(pixels);aligned=align_upper_shoulders(image,p,after)
        self.assertIsNot(aligned,image)
        self.assertEqual(aligned.getpixel((150,40)),image.getpixel((150,40)))
        self.assertEqual(aligned.getpixel((150,398)),image.getpixel((150,398)))
        self.assertAlmostEqual(aligned.getpixel((150,80))[0],95,delta=1)
        after[11].y=after[12].y=.4
        self.assertIs(align_upper_shoulders(image,p,after),image)

    def test_crop_uses_visible_frame_not_guessed_hips_and_still_rejects_shrinking(self):
        from PIL import Image
        p=self.points();p[0].y=.1;p[9].y=p[10].y=.1
        for i in (23,24):p[i].y=2.;p[i].visibility=p[i].presence=.01
        shoulder,bottom=torso_anchors(p,(300,400))
        self.assertEqual(bottom[1],399)
        source=np.zeros((400,300),np.float32);source[60:,70:230]=1
        skinny=np.zeros_like(source);skinny[60:,110:190]=1
        self.assertTrue(body_width_difference(Image.fromarray(source),Image.fromarray(source),p,(300,400))['passed'])
        self.assertFalse(body_width_difference(Image.fromarray(source),Image.fromarray(skinny),p,(300,400))['passed'])
        after=copy.deepcopy(p);after[11].y=after[12].y=.24
        self.assertTrue(pose_difference(p,after,(300,400),'upper')['neck_changed'])

    def points(self):
        p=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        p[11].x=.3;p[12].x=.7;p[11].y=p[12].y=.2;p[23].y=p[24].y=.6
        p[15].x=.45;p[15].y=.48
        for i in (17,19,21):p[i].x=.48;p[i].y=.52
        return p
    def test_protects_hand_not_entire_arm(self):
        mask=hand_protection(self.points(),(300,400))
        self.assertGreater(mask[200,140],0);self.assertEqual(mask[100,90],0)

    def test_upper_wrist_guard_extends_protection_near_cuff(self):
        points = self.points()
        normal = hand_protection(points,(300,400))
        guarded = hand_protection(points,(300,400),protect_wrist=True)
        self.assertEqual(normal[185,135],0)
        self.assertEqual(guarded[185,135],255)
        self.assertEqual(guarded[160,135],0)

    def test_full_sleeve_hand_guard_stops_at_wrist_plane(self):
        points=self.points()
        points[13].x=.45;points[13].y=.35
        guarded=hand_protection(points,(300,400),protect_wrist=True,clip_cuff=True)
        self.assertEqual(guarded[185,135],0)
        self.assertGreater(guarded[200,140],0)
        # An occluded elbow cannot establish a reliable cuff direction.
        points[13].visibility=.1
        self.assertGreater(hand_protection(points,(300,400),protect_wrist=True,clip_cuff=True)[190,135],0)
    def test_occluded_landmarks_not_guessed(self):
        p=self.points()
        for i in (15,16):p[i].visibility=0
        self.assertEqual(hand_protection(p,(300,400)).sum(),0)
    def test_identical_pose_passes(self):
        p=self.points();self.assertTrue(pose_difference(p,p,(300,400),'overall')['passed'])

    def test_neck_stretch_rejected_before_large_joint_displacement(self):
        points=self.points()
        points[9].y=points[10].y=.13
        after=copy.deepcopy(points)
        after[11].y=after[12].y=.25
        report=pose_difference(points,after,(300,400),'upper')
        self.assertEqual(report['shifted'],[])
        self.assertTrue(report['neck_changed'])
        self.assertFalse(report['passed'])

    def test_neck_guard_preserves_skin_without_preserving_shirt(self):
        labels=np.zeros((400,300),dtype=np.uint8)
        labels[50:85,140:160]=14
        labels[85:140,100:200]=5
        mask=neck_protection(self.points(),(300,400),labels)
        self.assertEqual(mask[75,150],255)
        self.assertEqual(mask[110,150],0)
        points=self.points();points[11].visibility=0
        self.assertEqual(neck_protection(points,(300,400),labels).sum(),0)

    def test_subtle_neck_extension_rejected_but_landmark_noise_allowed(self):
        points=self.points()
        points[9].y=points[10].y=.10
        after=copy.deepcopy(points)
        after[11].y=after[12].y=.216
        self.assertTrue(pose_difference(points,after,(300,400),'upper')['neck_changed'])
        after[11].y=after[12].y=.204
        self.assertTrue(pose_difference(points,after,(300,400),'upper')['passed'])

    def test_small_shoulder_drop_aligns_with_fixed_face_and_hips(self):
        from PIL import Image
        points=self.points();points[9].y=points[10].y=.10
        after=copy.deepcopy(points);after[11].y=after[12].y=.216
        pixels=np.zeros((400,300,3),dtype=np.uint8)
        pixels[:]=np.arange(400,dtype=np.uint16)[:,None,None]%256
        image=Image.fromarray(pixels)
        aligned=align_upper_shoulders(image,points,after)
        self.assertEqual(aligned.getpixel((150,40)),image.getpixel((150,40)))
        self.assertEqual(aligned.getpixel((150,240)),image.getpixel((150,240)))
        self.assertAlmostEqual(aligned.getpixel((150,80))[0],86,delta=1)

    def test_alignment_does_not_hide_large_or_uncertain_pose_changes(self):
        from PIL import Image
        image=Image.new('RGB',(300,400))
        points=self.points();points[9].y=points[10].y=.10
        after=copy.deepcopy(points);after[11].y=after[12].y=.28
        self.assertIs(align_upper_shoulders(image,points,after),image)
        after[11].y=after[12].y=.216;after[12].visibility=.1
        self.assertIs(align_upper_shoulders(image,points,after),image)

    def test_body_width_guard_detects_shrinking_with_identical_joints(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32)
        source[50:300,90:210]=1
        narrow=np.zeros_like(source);narrow[50:300,120:180]=1
        report=body_width_difference(Image.fromarray(source),Image.fromarray(narrow),self.points(),(300,400))
        self.assertTrue(report['available'])
        self.assertFalse(report['passed'])
        self.assertTrue(body_width_difference(Image.fromarray(source),Image.fromarray(source),self.points(),(300,400))['passed'])

    def test_body_width_does_not_use_scene_people_for_width(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32);source[50:300,90:210]=1
        other=source.copy();other[50:300,5:45]=1
        report=body_width_difference(Image.fromarray(source),Image.fromarray(other),self.points(),(300,400))
        self.assertEqual(report['ratios'],[1.,1.,1.,1.])

    def test_single_pixel_segmentation_hole_does_not_shrink_body(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32);source[50:300,90:210]=1
        output=source.copy();output[136,150]=output[160,150]=0
        report=body_width_difference(Image.fromarray(source),Image.fromarray(output),self.points(),(300,400))
        self.assertTrue(report['passed'])
        self.assertEqual(report['ratios'],[1.,1.,1.,1.])
    def test_changed_wrist_and_ankle_rejected(self):
        for index in (15,27):
            p=self.points();after=copy.deepcopy(p);after[index].x+=.2
            self.assertFalse(pose_difference(p,after,(300,400),'overall')['passed'])
    def test_missing_visible_output_joint_rejected(self):
        p=self.points();after=copy.deepcopy(p);after[15].presence=0
        self.assertIn(15,pose_difference(p,after,(300,400),'upper')['missing'])

    def test_uncertain_ankle_does_not_override_reliable_wrist(self):
        p=self.points();p[27].visibility=.54;after=copy.deepcopy(p);after[27].x+=.2
        self.assertTrue(pose_difference(p,after,(300,400),'overall')['passed'])
        after[15].x+=.2
        self.assertFalse(pose_difference(p,after,(300,400),'overall')['passed'])

    def test_footwear_edge_protection_does_not_cover_trousers(self):
        labels=np.zeros((400,300),dtype=np.uint8);labels[365:380,140:160]=18
        mask=foot_protection(None,(300,400),labels)
        self.assertGreater(mask[363,150],0)
        self.assertEqual(mask[320,150],0)

    def test_unreliable_foot_landmarks_do_not_create_protection(self):
        p=self.points()
        for i in (29,30,31,32):p[i].visibility=.1
        self.assertEqual(foot_protection(p,(300,400),np.zeros((400,300),dtype=np.uint8)).sum(),0)
