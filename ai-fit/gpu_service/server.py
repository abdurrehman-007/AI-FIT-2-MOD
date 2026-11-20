"""Run inside the official CatVTON environment. No synthetic inference fallback."""
import io
import os
import secrets
import threading
from flask import Flask,request,jsonify,send_file
from PIL import Image,ImageOps

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=24*1024*1024
lock=threading.Lock()
pipeline=None
masker=None
pose=None
segmentation=None


def initialize():
    global pipeline,masker,pose,segmentation
    import torch
    import mediapipe as mp
    from huggingface_hub import snapshot_download
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from rembg import new_session
    if not torch.cuda.is_available(): raise RuntimeError('CatVTON requires a configured CUDA GPU.')
    model_path=os.environ['POSE_MODEL_PATH']
    pose=mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model_path),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1))
    segmentation=new_session('u2netp')
    repo=snapshot_download(repo_id='zhengchong/CatVTON')
    masker=AutoMasker(densepose_ckpt=os.path.join(repo,'DensePose'),schp_ckpt=os.path.join(repo,'SCHP'),device='cuda')
    pipeline=CatVTONPipeline(base_ckpt=os.getenv('BASE_MODEL','runwayml/stable-diffusion-inpainting'),attn_ckpt=repo,attn_ckpt_version='mix',weight_dtype=torch.float16,use_tf32=True,device='cuda')


@app.get('/health')
def health():
    return jsonify(status='ready' if pipeline else 'not_initialized'),200 if pipeline else 503


@app.post('/infer')
def infer():
    key=os.getenv('CATVTON_API_KEY','')
    if not key or not secrets.compare_digest(request.headers.get('Authorization',''),'Bearer '+key): return jsonify(error='Unauthorized'),401
    if request.form.get('category') not in ('upper','lower','overall'): return jsonify(error='Invalid category'),400
    if not lock.acquire(blocking=False): return jsonify(error='GPU is busy; retry shortly.'),429
    try:
        if pipeline is None: initialize()
        import numpy as np
        import mediapipe as mp
        import torch
        from rembg import remove
        from utils import resize_and_crop,resize_and_padding
        from diffusers.image_processor import VaeImageProcessor
        images=[]
        for name in ('person','garment'):
            uploaded=request.files.get(name)
            if not uploaded: return jsonify(error='Person and garment images are required'),400
            im=Image.open(uploaded.stream)
            im.load()
            if im.format not in ('JPEG','PNG') or im.width*im.height>20_000_000: return jsonify(error='Invalid image'),400
            images.append(ImageOps.exif_transpose(im).convert('RGB'))
        person,garment=images
        cutout=remove(garment,session=segmentation).convert('RGBA')
        garment=Image.new('RGB',cutout.size,'white')
        garment.paste(cutout,mask=cutout.getchannel('A'))
        detected=pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(person)))
        if not detected.pose_landmarks or len(detected.pose_landmarks[0])!=33: return jsonify(error='No clear body pose detected'),422
        landmarks=detected.pose_landmarks[0]
        if any(landmarks[i].visibility<.4 for i in (11,12,23,24)): return jsonify(error='Keep shoulders and hips visible'),422
        person=resize_and_crop(person,(768,1024)); garment=resize_and_padding(garment,(768,1024))
        mask=masker(person,request.form['category'])['mask']
        processor=VaeImageProcessor(vae_scale_factor=8,do_normalize=False,do_binarize=True,do_convert_grayscale=True)
        mask=processor.blur(mask,blur_factor=9)
        with torch.inference_mode():
            output=pipeline(image=person,condition_image=garment,mask=mask,num_inference_steps=50,guidance_scale=2.5,generator=torch.Generator(device='cuda').manual_seed(secrets.randbelow(2**31)))[0]
        stream=io.BytesIO();output.save(stream,format='JPEG',quality=95);stream.seek(0)
        return send_file(stream,mimetype='image/jpeg')
    except (KeyError,ValueError,OSError):
        return jsonify(error='Invalid input or model configuration'),400
    except Exception:
        app.logger.exception('Inference failed')
        return jsonify(error='Inference failed'),503
    finally:
        lock.release()


if __name__=='__main__':
    if not os.getenv('CATVTON_API_KEY'): raise RuntimeError('Set CATVTON_API_KEY before starting.')
    initialize()
    app.run(host='127.0.0.1',port=8001,threaded=True)
