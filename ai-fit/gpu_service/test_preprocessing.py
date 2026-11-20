import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.preprocess-deps'))
from preprocessing import PhotoPreprocessor,InputRejected,validate_landmarks
class PreprocessingTests(unittest.TestCase):
    def test_upper_crop_accepts_only_strong_visible_torso_and_keeps_lower_requirements(self):
        p=self.points();p[0].y=.15
        for i in (23,24):p[i].y=1.2;p[i].visibility=p[i].presence=.01
        self.assertIs(validate_landmarks([p],'upper',(300,400)),p)
        for category in ('lower','overall'):
            with self.assertRaises(InputRejected):validate_landmarks([p],category,(300,400))
        p[12].visibility=.5
        with self.assertRaises(InputRejected):validate_landmarks([p],'upper',(300,400))
        p[12].visibility=1;p[11].y=p[12].y=.8
        with self.assertRaises(InputRejected):validate_landmarks([p],'upper',(300,400))

    def test_cropped_donor_model_is_detected_without_visible_hips(self):
        from unittest.mock import Mock
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        prep.mp=Mock();prep.pose=Mock()
        points=self.points()
        points[23].visibility=points[24].visibility=0
        prep.pose.detect.return_value=SimpleNamespace(pose_landmarks=[points])
        self.assertTrue(prep.garment_has_person(Image.new('RGB',(300,400))))
        points[0].visibility=.1
        self.assertFalse(prep.garment_has_person(Image.new('RGB',(300,400))))

    def points(self):
        p=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        p[11].x=.3;p[12].x=.7;p[11].y=p[12].y=.2;p[23].y=p[24].y=.6
        return p
    def test_valid_pose(self):
        for category in ('upper','lower','overall'):validate_landmarks([self.points()],category)

    def test_face_outside_photo_is_rejected(self):
        points=self.points();points[0].y=-.1
        with self.assertRaisesRegex(InputRejected,'face'):
            validate_landmarks([points],'upper')

    def test_input_boundary_is_required_for_structure_validation(self):
        from unittest.mock import Mock
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        prep.mp=Mock();prep.pose=Mock()
        prep.pose.detect.return_value=SimpleNamespace(pose_landmarks=[self.points()],segmentation_masks=[])
        with self.assertRaisesRegex(InputRejected,'outline'):
            prep.validate_person(Image.new('RGB',(300,400)),'upper',include_foreground=True)

    def test_missing_output_boundary_cannot_skip_body_check(self):
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        with patch.object(prep,'validate_person',return_value=(self.points(),None)):
            with self.assertRaisesRegex(InputRejected,'outline'):
                prep.check_output_pose(self.points(),Image.new('RGB',(300,400)),
                    'upper',original_foreground=Image.new('F',(300,400),1.))

    def test_insufficient_boundary_samples_are_rejected(self):
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        source=np.zeros((400,300),dtype=np.float32);source[130:145,70:230]=1
        with patch.object(prep,'validate_person',return_value=(self.points(),Image.fromarray(source))):
            with self.assertRaisesRegex(InputRejected,'outline'):
                prep.check_output_pose(self.points(),Image.new('RGB',(300,400)),
                    'upper',original_foreground=Image.fromarray(source))

    def test_output_body_width_is_checked_even_when_joints_match(self):
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        source=np.zeros((400,300),dtype=np.float32);source[40:300,70:230]=1
        output=np.zeros_like(source);output[40:300,110:190]=1
        with patch.object(prep,'validate_person',return_value=(self.points(),Image.fromarray(output))):
            with self.assertRaisesRegex(InputRejected,'body width'):
                prep.check_output_pose(self.points(),Image.new('RGB',(300,400)),
                    'upper',original_foreground=Image.fromarray(source))
    def test_pose_width_is_measured_in_pixels(self):
        points=self.points()
        points[11].x=.45;points[12].x=.55
        # Same normalized points have very different body proportions in a
        # narrow portrait and a wide scene. Do not compare x/y units directly.
        with self.assertRaises(InputRejected):
            validate_landmarks([points],'upper',(300,1200))
        validate_landmarks([points],'upper',(1200,600))

    def test_garment_margins_are_trimmed_with_silhouette_retained(self):
        prep=PhotoPreprocessor.__new__(PhotoPreprocessor)
        photo=Image.new('RGBA',(400,400),(0,0,0,0))
        photo.paste((50,70,100,255),(140,60,260,340))
        result=prep.clean_garment(photo)
        self.assertLess(result.width,160)
        self.assertLess(result.height,310)
        self.assertEqual(result.info['garment_silhouette'].size,result.size)
    def test_invalid_poses(self):
        for poses in ([],[self.points(),self.points()],[self.points()[:20]]):
            with self.assertRaises(InputRejected):validate_landmarks(poses,'upper')
    def test_missing_and_nonfinite_landmarks(self):
        for value in (0.,float('nan')):
            p=self.points();p[11].visibility=value
            with self.assertRaises(InputRejected):validate_landmarks([p],'upper')
    def test_category_and_lower_visibility(self):
        p=self.points();p[28].presence=0
        validate_landmarks([p],'upper')
        for category in ('lower','overall','bad'):
            with self.assertRaises(InputRejected):validate_landmarks([p],category)
    def test_existing_alpha_preserved(self):
        p=PhotoPreprocessor.__new__(PhotoPreprocessor)
        im=Image.new('RGBA',(128,128),(0,0,0,0));im.paste((20,40,200,255),(32,32,96,96))
        with patch('rembg.remove') as remove:
            out=p.clean_garment(im);remove.assert_not_called()
        self.assertEqual(out.getpixel((0,0)),(255,255,255));self.assertEqual(out.getpixel((out.width//2,out.height//2)),(20,40,200))
    def test_empty_garments_rejected(self):
        p=PhotoPreprocessor.__new__(PhotoPreprocessor)
        for im in (Image.new('RGB',(128,128),'white'),Image.new('RGBA',(128,128),(0,0,0,0)),Image.new('RGB',(32,32))):
            with self.assertRaises(InputRejected):p.clean_garment(im)
if __name__=='__main__':unittest.main()
