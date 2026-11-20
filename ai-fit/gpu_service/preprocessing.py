"""Local pose quality checks and garment foreground extraction for still photos."""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from pose_geometry import visible, pose_difference, body_width_difference, align_upper_shoulders, cropped_upper


class InputRejected(ValueError):
    pass


class SleeveRejected(InputRejected):
    """A candidate contradicts the reference sleeve length."""


class LengthRejected(InputRejected):
    """A declared long upper garment was shortened in generation."""


def validate_landmarks(poses, category, size=None):
    if category not in ('upper', 'lower', 'overall'):
        raise InputRejected('Invalid category')
    if not poses:
        raise InputRejected('No clear person detected. Use a well-lit photo facing the camera.')
    if len(poses) != 1:
        raise InputRejected('Use a photo with only one person in it.')
    points = poses[0]
    if len(points) != 33:
        raise InputRejected('Body pose could not be read. Choose another photo.')
    if not visible(points[0],.4):
        raise InputRejected('Keep your face clearly visible in the photo so your head and neck can be preserved.')
    crop=category=='upper' and cropped_upper(points,size or (1,1))
    required = ((11,12) if crop else (11,12,23,24)) if category=='upper' else (11,12,23,24,25,26,27,28)
    for i in required:
        p = points[i]
        if (not np.isfinite([p.x, p.y, p.visibility, p.presence]).all()
                or p.visibility < .4 or p.presence < .4
                or not (0 <= p.x <= 1 and 0 <= p.y <= 1)):
            region = 'shoulders and hips' if category == 'upper' else 'shoulders, hips, knees and ankles'
            raise InputRejected(f'Keep your {region} clearly visible inside the photo.')
    width, height = size or (1, 1)
    shoulders = np.hypot((points[11].x-points[12].x)*width,
                         (points[11].y-points[12].y)*height)
    torso = (1-(points[11].y+points[12].y)/2) if crop else (points[23].y+points[24].y-points[11].y-points[12].y)/2
    if torso < .12 or shoulders < width*.07 or shoulders < torso*height*.3:
        raise InputRejected('Stand upright facing the camera and move closer so your torso is clear.')
    return points


class PhotoPreprocessor:
    def __init__(self, folder):
        folder = Path(folder).resolve()
        for name in ('pose_landmarker_full.task', 'u2netp.onnx'):
            if not (folder/name).is_file():
                raise RuntimeError('Preprocessing models are missing. Run gpu_service/setup_preprocessing.py.')
        import mediapipe as mp
        from rembg import new_session
        self.mp = mp
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=(folder/'pose_landmarker_full.task').read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=2,
                output_segmentation_masks=True))
        os.environ['U2NET_HOME'] = str(folder)
        try:
            self.segmentation = new_session('u2netp', providers=['CPUExecutionProvider'])
        except Exception:
            self.pose.close()
            raise

    def validate_person(self, image, category, include_foreground=False):
        if min(image.size) < 128:
            raise InputRejected('Your photo is too small. Upload a larger, clear photo.')
        sample = ImageOps.contain(image.convert('RGB'), (1280, 1280))
        detected = self.pose.detect(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=np.array(sample)))
        points=validate_landmarks(detected.pose_landmarks, category, image.size)
        if include_foreground:
            foreground=None
            if detected.segmentation_masks:
                probabilities=detected.segmentation_masks[0].numpy_view().copy().squeeze()
                foreground=Image.fromarray(probabilities).resize(image.size,Image.Resampling.BILINEAR)
            if foreground is None:
                raise InputRejected('Your body outline could not be read clearly. Use a photo with a clear outline against the background.')
            if category=='upper' and cropped_upper(points,image.size):
                boundary=np.asarray(foreground)[-max(1,round(image.height*.01)):]
                center=round((points[11].x+points[12].x)*image.width/2)
                radius=max(2,round(abs(points[11].x-points[12].x)*image.width*.25))
                if np.mean(boundary[:,max(0,center-radius):min(image.width,center+radius)]>.5)<.6:
                    raise InputRejected('Your torso could not be verified at the photo boundary. Include more of your upper body.')
            return points,foreground
        return points

    def garment_has_person(self, image):
        sample=ImageOps.contain(image.convert('RGB'),(1280,1280))
        detected=self.pose.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.array(sample)))
        # A cropped model photo can show a face and both shoulders without
        # visible hips. It still needs donor removal before conditioning.
        return any(len(points)==33 and all(visible(points[i],.65) for i in (0,11,12))
                   for points in detected.pose_landmarks)

    def align_upper_shoulders(self, original_points, output, neck_anchor='mouth'):
        try:
            generated_points=self.validate_person(output,'upper')
        except InputRejected as error:
            raise InputRejected('The generated pose could not be verified. This result was not saved; try another garment photo.') from error
        return align_upper_shoulders(output,original_points,generated_points,neck_anchor=neck_anchor)

    def check_output_pose(self, original_points, output, category, original_foreground=None, neck_anchor='mouth'):
        try:
            output_points, output_foreground=self.validate_person(output,category,include_foreground=True)
        except InputRejected as error:
            raise InputRejected('The generated pose could not be verified. Try a garment-only photo; this result was not saved.') from error
        report=pose_difference(original_points,output_points,output.size,category,neck_anchor=neck_anchor)
        if report.get('neck_changed'):
            raise InputRejected('The generated preview stretched the neck or moved the shoulders too much. This result was not saved. Try a different garment photo.')
        if not report['passed']:
            raise InputRejected('The outfit changed your pose too much. Try a garment-only product photo or a clearer pose; this result was not saved.')
        if original_foreground is not None and output_foreground is not None:
            shape=body_width_difference(original_foreground,output_foreground,original_points,output.size)
            report['body_shape']=shape
            if not shape['available']:
                raise InputRejected('The generated body outline could not be verified. This result was not saved. Try a clearer photo.')
            if not shape['passed']:
                raise InputRejected('The generated preview changed your body width too much. This result was not saved. Try a different garment photo.')
        elif original_foreground is not None:
            raise InputRejected('The generated body outline could not be verified. This result was not saved. Try a clearer photo.')
        return report

    def clean_garment(self, image):
        from rembg import remove
        if min(image.size) < 64:
            raise InputRejected('The garment photo is too small. Use a clearer product photo.')
        sample = image.convert('RGBA')
        sample.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        # Honour an existing cutout instead of replacing its supplied transparency.
        alpha = np.asarray(sample.getchannel('A'))
        if not np.any(alpha < 250):
            if np.asarray(sample.convert('RGB')).std(axis=(0, 1)).max() < 2:
                raise InputRejected('No garment found. Upload a clear photo of one garment.')
            sample = remove(sample, session=self.segmentation).convert('RGBA')
        alpha = np.asarray(sample.getchannel('A'))
        if np.mean(alpha > 127) < .025:
            raise InputRejected('The garment could not be separated. Use one garment on a plain contrasting background.')
        result = Image.new('RGB', sample.size, 'white')
        result.paste(sample, mask=sample.getchannel('A'))
        # Product-photo margins otherwise reduce the garment to a small patch
        # in the conditioning image, particularly for jeans and fitted shirts.
        ys, xs = np.where(alpha > 127)
        bounds = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
        margin = max(2, round(max(bounds[2]-bounds[0], bounds[3]-bounds[1])*.03))
        box = (max(0,bounds[0]-margin), max(0,bounds[1]-margin),
               min(result.width,bounds[2]+margin), min(result.height,bounds[3]+margin))
        result = result.crop(box)
        result.info['garment_silhouette'] = Image.fromarray(alpha).crop(box)
        return result

    def close(self):
        self.pose.close()
