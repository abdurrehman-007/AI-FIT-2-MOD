"""Local Windows try-on service with FASHN, DeCo and protected compositing."""
import os
import sys
import io
import secrets
import threading
import re
import json
import gc
import hashlib
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('AIFIT_EXISTING_PROJECT', r'E:\Comsats\8th Semester\Fyp 2\Virtual-Try-On-main'))
DECO_CHECKPOINT = ROOT / '.models' / 'deco'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(ROOT / '.preprocess-deps'))
sys.path.insert(0, str(PROJECT / 'CatVTON'))
sys.path.insert(0, str(PROJECT / 'ai_engine' / 'clothes'))
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, request, jsonify, send_file, g
from huggingface_hub import snapshot_download
from clothing_mask import ClothingMask, restore_crop
from preprocessing import PhotoPreprocessor, InputRejected, SleeveRejected, LengthRejected
from pose_geometry import cropped_upper
from runtime_fingerprint import code_fingerprint, runtime_fingerprint
from progress import Progress
from fast_attention import attention_stats

LOADED_CODE_FINGERPRINT=code_fingerprint(ROOT)

app = Flask(__name__)
progress=Progress()
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024
lock = threading.Lock()
admission = threading.Condition()
foreground_waiting = 0
pipeline = None
clothing_mask = None
preprocessor = None
backend_name = None
infer_size = (768, 1024)
cached_renderer = None
keypath = ROOT / 'instance' / 'gpu.key'
keypath.parent.mkdir(exist_ok=True)
if not keypath.exists():
    with keypath.open('x') as f:
        f.write(secrets.token_urlsafe(40))
KEY = keypath.read_text().strip()


def _deco_paths():
    vendor = str(SERVICE / 'vendor')
    for path in (vendor,):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_clothing_mask():
    global clothing_mask
    checkpoint = snapshot_download('zhengchong/CatVTON', local_files_only=True)
    clothing_mask = ClothingMask(Path(checkpoint) / 'SCHP/exp-schp-201908261155-lip.pth')
    return checkpoint


def _load_deco():
    global pipeline, backend_name, infer_size
    weights = DECO_CHECKPOINT / 'unet' / 'diffusion_pytorch_model.safetensors'
    if not weights.exists():
        raise RuntimeError('DeCo-VTON weights are missing. Run gpu_service/download_deco.py first.')
    original_path = list(sys.path)
    try:
        _deco_paths()
        from deco_backend import DeCoBackend
        pipeline = DeCoBackend(DECO_CHECKPOINT)
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        backend_name = 'deco'
        infer_size = pipeline.resolution
        print(f'Model ready: DeCo-VTON {infer_size[0]} x {infer_size[1]}, 50 steps, safety checker enabled.', flush=True)
        return True
    except Exception:
        sys.path[:] = original_path
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        raise


def _load_fashn():
    global pipeline,backend_name,infer_size
    from setup_fashn import FILES,valid
    checkpoint=ROOT/'.models/fashn'
    for relative,(_,_,size,expected) in FILES.items():
        if not valid(checkpoint/relative,size,expected):
            raise RuntimeError('FASHN assets are missing or invalid. Run gpu_service/setup_fashn.py.')
    from fashn_backend import FashnBackend
    pipeline=FashnBackend(checkpoint)
    backend_name='fashn';infer_size=pipeline.resolution
    print('Model ready: FASHN 576 x 864, sequential guidance, safety checker enabled.',flush=True)


def select_engine():
    engine=os.environ.get('AIFIT_ENGINE','auto')
    if engine!='auto':return engine
    required=[ROOT/'.models/fashn'/name for name in (
        'model.safetensors','dwpose/yolox_l.onnx','dwpose/dw-ll_ucoco_384.onnx',
        'parser/config.json','parser/model.safetensors')]
    required.extend((ROOT/'.external/fashn-vton-1.5/src/fashn_vton/pipeline.py',
        ROOT/'.runtime-deps/fashn/fashn_human_parser/__init__.py',ROOT/'.runtime-deps/fashn/einops/__init__.py'))
    return 'fashn' if all(path.is_file() for path in required) else 'deco'


def garment_signature(image):
    image=image.convert('RGBA')
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def preferred_engine(image, default):
    path=ROOT/'instance/garment-backends.json'
    if not path.is_file():return default
    try:
        entry=json.loads(path.read_text(encoding='utf-8')).get('garments',{}).get(garment_signature(image),{})
        engine=entry.get('engine')
        return engine if engine in ('fashn','deco') else default
    except (OSError,ValueError,AttributeError):
        app.logger.warning('Garment backend calibration could not be read.')
        return default


def deco_available():
    return (DECO_CHECKPOINT/'unet/diffusion_pytorch_model.safetensors').is_file()


def requested_engine(image, category, explicit='auto'):
    if explicit!='auto':return explicit
    engine=select_engine()
    if os.environ.get('AIFIT_ENGINE','auto')=='auto':
        # This repair changes upper rendering. Retain the established engine
        # for the other categories instead of silently changing their output.
        if category!='upper' and deco_available():return 'deco'
        engine=preferred_engine(image,engine)
    return engine


def image_engine(reference, category, explicit, upper_length, person, landmarks, garment):
    engine=requested_engine(reference,category,explicit)
    if (engine=='fashn' and category=='upper' and upper_length=='standard'
            and explicit=='auto' and os.environ.get('AIFIT_ENGINE','auto')=='auto'
            and preferred_engine(reference,'auto')=='auto' and deco_available()
            and clothing_mask.prefer_preserved_upper(person,landmarks,garment) is True):
        return 'deco'
    return engine


def generation_steps(fashn, category, upper_length, attempt, short_source=False):
    # Local fast candidates use 12/15 steps; balanced retains the upstream
    # 20-step profile. Long outfits and retries keep their quality settings.
    standard=category=='upper' and upper_length=='standard' and attempt==1
    fast=os.environ.get('AIFIT_SPEED_PROFILE','balanced')=='fast'
    if not fashn:return 35 if fast and standard else 50
    return ((15 if short_source else 12) if fast else 20) if standard else 30


def can_park_renderer(renderer):
    try:
        import psutil
        estimate=getattr(renderer,'park_bytes',None)
        needed=estimate() if callable(estimate) else renderer.resident_bytes()
        return (callable(getattr(renderer,'park_cpu',None))
            and callable(getattr(renderer,'activate_cuda',None))
            and psutil.virtual_memory().available>needed+1.5*1024**3)
    except (ImportError,AttributeError,TypeError):
        return False


def activate_engine(engine):
    """Keep one renderer on CUDA and at most one inactive renderer in RAM."""
    global pipeline,backend_name,infer_size,cached_renderer
    if pipeline is not None and backend_name==engine:return
    if engine=='deco' and not deco_available():
        raise ValueError('The calibrated DeCo renderer is not installed.')
    loaders={'fashn':_load_fashn,'deco':_load_deco,
             'catvton':lambda:_load_catvton(snapshot_download('zhengchong/CatVTON',local_files_only=True))}
    if engine not in loaders:raise ValueError('Invalid inference engine.')
    previous=backend_name
    target=cached_renderer if cached_renderer is not None and cached_renderer[0]==engine else None
    cached_renderer=None
    if pipeline is not None and can_park_renderer(pipeline):
        try:
            pipeline.park_cpu()
            cached_renderer=(previous,pipeline,infer_size)
        except Exception:
            app.logger.exception('Renderer RAM parking failed; using a cold switch.')
    pipeline=None;backend_name=None
    if target is None or cached_renderer is None:gc.collect()
    torch.cuda.empty_cache()
    failure=None
    try:
        if target is not None:
            target[1].activate_cuda()
            backend_name,pipeline,infer_size=target
        else:
            loaders[engine]()
    except Exception as error:
        failure=str(error)
    if failure is not None:
        pipeline=None;backend_name=None
        target=None
        gc.collect();torch.cuda.empty_cache()
        if previous in loaders and previous!=engine:
            try:
                if cached_renderer is not None and cached_renderer[0]==previous:
                    cached_renderer[1].activate_cuda()
                    backend_name,pipeline,infer_size=cached_renderer
                    cached_renderer=None
                else:loaders[previous]()
            except Exception:app.logger.exception('Previous renderer could not be restored.')
        raise RuntimeError('Renderer activation failed: '+failure)


def project_edit_mask(mask,box,protected,size):
    full=Image.new('L',size)
    full.paste(mask.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.BILINEAR),box[:2])
    full.paste(0,(0,0),protected)
    return full


def preview_foreground(output):
    try:
        return preprocessor.validate_person(output,'upper',include_foreground=True)[1]
    except InputRejected as error:
        raise InputRejected('The generated body outline could not be verified. This result was not saved; try another garment photo.') from error


def acquire_slot(background=False, timeout=90):
    """Bound interactive waiting and yield audit slots to waiting shoppers."""
    global foreground_waiting
    with admission:
        if background:
            return not foreground_waiting and lock.acquire(False)
        if lock.acquire(False):return True
        if foreground_waiting>=2:return False
        foreground_waiting+=1
        admission.notify_all()
    try:
        return lock.acquire(timeout=timeout)
    finally:
        with admission:
            foreground_waiting-=1
            admission.notify_all()


def _load_catvton(checkpoint):
    global pipeline, backend_name, infer_size
    from model.pipeline import CatVTONPipeline
    from safetensors.torch import load_file
    pipeline = CatVTONPipeline(
        base_ckpt='runwayml/stable-diffusion-inpainting', attn_ckpt=checkpoint,
        attn_ckpt_version='mix', weight_dtype=torch.float16, device='cuda', use_tf32=True)
    trained = load_file(str(Path(checkpoint) / 'mix-48k-1024/attention/model.safetensors'))
    current = pipeline.attn_modules.state_dict()
    unexpected = set(trained) - set(current)
    mismatches = [
        name for name in trained
        if name in current and not torch.equal(current[name].detach().cpu(), trained[name].to(dtype=current[name].dtype))
    ]
    aliases = dict(pipeline.attn_modules.named_parameters(remove_duplicate=False))
    covered = {id(aliases[name]) for name in trained if name in aliases}
    uncovered = {id(p) for p in pipeline.attn_modules.parameters()} - covered
    if uncovered or unexpected or mismatches:
        raise RuntimeError('CatVTON attention weights do not match the loaded pipeline.')
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    original_checker = pipeline.run_safety_checker

    def checked(*args, **kwargs):
        result, flags = original_checker(*args, **kwargs)
        pipeline.last_safety_blocked = bool(flags is not None and any(flags))
        return result, flags

    pipeline.run_safety_checker = checked
    backend_name = 'catvton'
    infer_size = (768, 1024)
    print('Model ready: CatVTON 768 x 1024, 50 steps, fp16, safety checker enabled.', flush=True)


def initialize():
    global clothing_mask, preprocessor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    print('Loading try-on weights on ' + torch.cuda.get_device_name(0), flush=True)
    torch.set_num_threads(4)
    preprocessor = PhotoPreprocessor(ROOT / '.models/preprocessing')
    checkpoint = _load_clothing_mask()
    engine = select_engine()
    if engine == 'fashn':
        _load_fashn()
    elif engine == 'deco':
        _load_deco()
    elif engine == 'catvton':
        _load_catvton(checkpoint)
    else:
        raise ValueError('AIFIT_ENGINE must be auto, fashn, deco or catvton')


@app.get('/health')
def health():
    return jsonify(
        ready=pipeline is not None and preprocessor is not None and clothing_mask is not None,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        backend=backend_name,
        resolution=list(infer_size),
        pose_validation=preprocessor is not None,
        pose_preservation="landmark-protected inpainting and output check",
        garment_segmentation=preprocessor is not None,
        preprocessing_version='upper-repair-v40',
        attention=attention_stats(),
        speed_profile=os.environ.get('AIFIT_SPEED_PROFILE','balanced'),
        cached_backend=cached_renderer[0] if cached_renderer is not None else None,
        source_fingerprint=runtime_fingerprint(ROOT,LOADED_CODE_FINGERPRINT),
        busy=lock.locked(),
        queued=foreground_waiting,
    ), 200 if pipeline is not None and preprocessor is not None and clothing_mask is not None else 503


@app.get('/progress/<key>')
def inference_progress(key):
    if not secrets.compare_digest(request.headers.get('Authorization',''),'Bearer '+KEY):
        return jsonify(error='Unauthorized'),401
    row=progress.snapshot(key)
    return (jsonify(row),200) if row is not None else (jsonify(error='Progress unavailable'),404)


@app.after_request
def finish_progress(response):
    if request.endpoint=='infer':progress.finish(getattr(g,'progress_key',None),response.status_code==200)
    return response


@app.post('/infer')
def infer():
    from deco_backend import OutputBlocked
    if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + KEY):
        return jsonify(error='Unauthorized'), 401
    category = request.form.get('category', 'upper')
    if category not in ('upper', 'lower', 'overall'):
        return jsonify(error='Invalid category'), 400
    upper_length=request.form.get('upper_length','standard')
    if upper_length not in ('standard','long'):
        return jsonify(error='Invalid upper garment length'),400
    progress_key=request.headers.get('X-AIFIT-Request-ID')
    if progress_key is not None and not re.fullmatch(r'[0-9a-f]{32}',progress_key):
        return jsonify(error='Invalid request identifier'),400
    try:progress.start(progress_key)
    except ValueError as error:return jsonify(error=str(error)),429
    g.progress_key=progress_key
    waiting_started=time.monotonic()
    background=bool(re.fullmatch(r'[0-9a-f]{24}',request.headers.get('X-AIFIT-Audit-Case','')))
    if not acquire_slot(background=background):
        return jsonify(error='The fitting room is busy. Retry shortly.'), 429
    queued_seconds=time.monotonic()-waiting_started
    timings={}
    processing_started=time.monotonic()
    try:
        if pipeline is None or preprocessor is None or clothing_mask is None:
            return jsonify(error='The try-on service is still starting. Retry shortly.'), 503
        clothing_mask.begin_request()
        progress.update(progress_key,'validating',4)
        images = []
        for name in ('person', 'garment'):
            file = request.files.get(name)
            if not file:
                return jsonify(error='Missing image'), 400
            im = Image.open(file.stream)
            if im.format not in ('JPEG', 'PNG') or im.width * im.height > 20_000_000:
                return jsonify(error='Invalid image'), 400
            im.load()
            images.append(ImageOps.exif_transpose(im).convert('RGBA' if name == 'garment' else 'RGB'))
        try:
            phase_started=time.monotonic()
            landmarks, foreground = preprocessor.validate_person(images[0], category, include_foreground=True)
            clean_garment = preprocessor.clean_garment(images[1])
            if preprocessor.garment_has_person(clean_garment):
                clean_garment = clothing_mask.garment_only(clean_garment, category)
            if backend_name in ('fashn','deco','catvton'):
                timings['input']=round(time.monotonic()-phase_started,3)
                phase_started=time.monotonic()
                progress.update(progress_key,'preparing',10)
                activate_engine(image_engine(images[1],category,request.form.get('engine','auto'),
                    upper_length,images[0],landmarks,clean_garment))
                timings['renderer']=round(time.monotonic()-phase_started,3)
                phase_started=time.monotonic()
            person, mask, crop_box, protected = clothing_mask.prepare(
                images[0], category, infer_size, landmarks=landmarks, include_protection=True,
                foreground=foreground, garment=clean_garment,upper_length=upper_length)
            neck_anchor=clothing_mask.neck_anchor(images[0],landmarks) if category=='upper' else 'mouth'
            steps=generation_steps(backend_name=='fashn',category,upper_length,1)
            progress.update(progress_key,'preparing',18,remaining=steps*(1.6 if backend_name=='fashn' else .45)+8)
            timings['prepare']=round(time.monotonic()-phase_started,3)
        except ValueError as error:
            return jsonify(error=str(error)), 422
        garment = ImageOps.pad(clean_garment, infer_size, method=Image.Resampling.LANCZOS, color='white')
        full_mask=project_edit_mask(mask,crop_box,protected,images[0].size)
        refinement_used=False
        fashn=backend_name=='fashn'
        short_source=category=='upper' and clothing_mask.short_source_upper(images[0],landmarks,clean_garment) is True
        masked_first=category=='upper' and fashn and (upper_length=='long' or clothing_mask.has_layered_upper(images[0]) is True)
        model_category='overall' if category=='upper' and upper_length=='long' else category
        use_deco_retry=False
        for attempt,seed in enumerate(((42,73) if masked_first else (73,42)) if fashn else (42,73),1):
            progress.update(progress_key,'generating' if attempt==1 else 'refining',20 if attempt==1 else 88,
                remaining=generation_steps(fashn,category,upper_length,attempt,short_source)*(1.6 if fashn else .45)+8,attempt=attempt)
            if use_deco_retry:
                activate_engine('deco');fashn=False;seed=73
                person,mask,crop_box,protected=clothing_mask.prepare(images[0],category,infer_size,
                    landmarks=landmarks,include_protection=True,foreground=foreground,garment=clean_garment,upper_length=upper_length)
                full_mask=project_edit_mask(mask,crop_box,protected,images[0].size)
                garment=ImageOps.pad(clean_garment,infer_size,method=Image.Resampling.LANCZOS,color='white')
            with torch.inference_mode():
                phase_started=time.monotonic()
                torch.manual_seed(seed)
                generator = torch.Generator(device='cuda').manual_seed(seed)
                output = pipeline(
                    image=person, condition_image=garment, mask=mask,
                    width=infer_size[0], height=infer_size[1],
                    num_inference_steps=generation_steps(fashn,category,upper_length,attempt,short_source),
                    progress_callback=progress.steps(progress_key,steps,20 if attempt==1 else 88,80 if attempt==1 else 94,
                        synchronize=torch.cuda.synchronize,initial_rate=1.6 if fashn else .45,
                        stage='generating' if attempt==1 else 'refining'),
                    guidance_scale=(2.0 if attempt==1 and not masked_first else 3.0) if fashn else (3.5 if category=='upper' else 2.5),
                    generator=generator,**({'category':model_category,'segmentation_free':attempt==1 and not masked_first} if fashn else {}))[0]
                timings['generation']=round(timings.get('generation',0)+time.monotonic()-phase_started,3)
            if getattr(pipeline, 'last_safety_blocked', False):
                raise OutputBlocked('The generated image could not be returned. Try a different photo or garment.')
            output = restore_crop(output, images[0], mask, crop_box, protected=protected)
            progress.update(progress_key,'checking',84 if attempt==1 else 95,remaining=8)
            independent_clothing=None
            try:
                if category == 'upper':
                    aligned = preprocessor.align_upper_shoulders(landmarks, output,neck_anchor=neck_anchor)
                    output = restore_crop(aligned.crop(crop_box), images[0], mask,
                                          crop_box, protected=protected)
                    output = clothing_mask.restore_exposed_skin(images[0], output,edit_mask=full_mask,garment=clean_garment)
                    output = clothing_mask.restore_plain_surface(output,clean_garment,edit_mask=full_mask,landmarks=landmarks)
                    if upper_length=='long':
                        parsed=pipeline.parse_preview(output) if fashn else None
                        independent_clothing=np.isin(parsed,(3,4,10)) if parsed is not None else None
                        output=clothing_mask.restore_lower_layers(images[0],output,landmarks,full_mask,
                            confirmed_lower=np.isin(parsed,(6,14,15)) if parsed is not None else None)
                    output_foreground=preview_foreground(output)
                    output = clothing_mask.restore_background_edges(images[0],output,foreground,full_mask,preview_foreground=output_foreground)
                preprocessor.check_output_pose(landmarks, output, category, original_foreground=foreground,neck_anchor=neck_anchor)
                if category == 'upper':
                    coverage=clothing_mask.check_sleeve_coverage(output,landmarks,clean_garment,source=images[0],edit_mask=full_mask)
                    if coverage.get('passed') is False and not refinement_used:
                        repair=clothing_mask.prepare_sleeve_refinement(output,landmarks,
                            coverage.get('short_arms',[])+coverage.get('long_arms',[]),infer_size,full_mask,protected)
                        if repair is not None:
                            progress.update(progress_key,'refining',88,remaining=(20 if fashn else 35)*(1.6 if fashn else .45)+8)
                            refinement_used=True
                            repair_person,repair_mask,repair_box=repair
                            with torch.inference_mode():
                                torch.manual_seed(seed+1000)
                                repaired=pipeline(image=repair_person,condition_image=garment,
                                    progress_callback=progress.steps(progress_key,20 if fashn else 35,88,94,synchronize=torch.cuda.synchronize,
                                        initial_rate=1.6 if fashn else .45,stage='refining'),
                                    mask=repair_mask,width=infer_size[0],height=infer_size[1],
                                    num_inference_steps=20 if fashn else 35,guidance_scale=2.5 if fashn else 4.5,
                                    generator=torch.Generator(device='cuda').manual_seed(seed+1000),
                                    **({'category':model_category,'segmentation_free':True} if fashn else {}))[0]
                            if getattr(pipeline,'last_safety_blocked',False):
                                raise OutputBlocked('The generated image could not be returned. Try a different photo or garment.')
                            output=restore_crop(repaired,output,repair_mask,repair_box,protected=protected)
                            output=clothing_mask.restore_exposed_skin(images[0],output,edit_mask=full_mask,garment=clean_garment)
                            output=clothing_mask.restore_plain_surface(output,clean_garment,edit_mask=full_mask,landmarks=landmarks)
                            if upper_length=='long':
                                parsed=pipeline.parse_preview(output) if fashn else None
                                independent_clothing=np.isin(parsed,(3,4,10)) if parsed is not None else None
                                output=clothing_mask.restore_lower_layers(images[0],output,landmarks,full_mask,
                                    confirmed_lower=np.isin(parsed,(6,14,15)) if parsed is not None else None)
                            output_foreground=preview_foreground(output)
                            output=clothing_mask.restore_background_edges(images[0],output,foreground,full_mask,preview_foreground=output_foreground)
                            preprocessor.check_output_pose(landmarks,output,category,original_foreground=foreground,neck_anchor=neck_anchor)
                            coverage=clothing_mask.check_sleeve_coverage(output,landmarks,clean_garment,source=images[0],edit_mask=full_mask)
                    if coverage.get('passed') is False:
                        raise SleeveRejected('The preview changed the sleeve length. This result was not saved. Try a different garment photo or a clearer view of your arms.')
                    surface=clothing_mask.check_upper_surface(output,landmarks,clean_garment,full_mask)
                    if surface.get('passed') is False:
                        raise InputRejected('The preview added a different inner layer or opened the garment. This result was not saved. Try another product photo.')
                    length=clothing_mask.check_upper_length(output,landmarks,upper_length,full_mask,independent_clothing=independent_clothing)
                    if length.get('passed') is False:
                        raise LengthRejected('The preview shortened this long garment. This result was not saved; try another garment photo.')
            except InputRejected as error:
                # Authenticated audit requests can retain failed candidates
                # privately for visual diagnosis, never as public previews.
                audit_case=request.headers.get('X-AIFIT-Audit-Case','')
                if re.fullmatch(r'[0-9a-f]{24}',audit_case):
                    diagnostic=ROOT/'instance/upper-audit-diagnostics'
                    diagnostic.mkdir(exist_ok=True)
                    output.save(diagnostic/f'{audit_case}-{attempt}.jpg',quality=95)
                    (diagnostic/f'{audit_case}-{attempt}.json').write_text(
                        json.dumps({'error':str(error),'attempt':attempt}),encoding='utf-8')
                if attempt == 2:
                    raise
                use_deco_retry=(fashn and (isinstance(error,SleeveRejected) or (not isinstance(error,LengthRejected) and cropped_upper(landmarks,images[0].size))) and deco_available()
                    and request.form.get('engine','auto')=='auto'
                    and os.environ.get('AIFIT_ENGINE','auto')=='auto')
                app.logger.info('Output structure check failed; trying the second candidate.')
                continue
            break
        result = io.BytesIO()
        progress.update(progress_key,'saving',98,remaining=2)
        output.save(result, format='JPEG', quality=95)
        result.seek(0)
        response=send_file(result, mimetype='image/jpeg')
        response.headers['X-AIFIT-Attempts']=str(attempt)
        response.headers['X-AIFIT-Sleeve-Refinement']='1' if refinement_used else '0'
        response.headers['X-AIFIT-Backend']=backend_name or 'unknown'
        response.headers['X-AIFIT-Queue-Wait']=f'{queued_seconds:.2f}'
        response.headers['X-AIFIT-Steps']=str(generation_steps(fashn,category,upper_length,attempt,short_source))
        response.headers['X-AIFIT-Attention']=json.dumps(attention_stats())
        timings['total']=round(time.monotonic()-processing_started,3)
        stats=getattr(clothing_mask,'label_stats',None)
        if isinstance(stats,dict):timings['segmentation']=stats
        response.headers['X-AIFIT-Timings']=json.dumps(timings,separators=(',',':'))
        app.logger.info('Inference timing %s',timings)
        return response
    except (OutputBlocked, InputRejected) as error:
        return jsonify(error=str(error)), 422
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error='The image could not be read. Upload a valid JPEG or PNG.'), 400
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify(error='GPU memory exhausted. Close other GPU applications.'), 503
    except RuntimeError as error:
        # Some CUDA kernels report allocation failure as RuntimeError rather
        # than torch.cuda.OutOfMemoryError. Keep other failures as HTTP 500.
        message = str(error).lower()
        if 'cuda' in message and 'out of memory' in message:
            torch.cuda.empty_cache()
            return jsonify(error='GPU memory exhausted. Close other GPU applications and retry.'), 503
        app.logger.exception('Inference runtime failed')
        return jsonify(error='Inference failed; inspect local GPU log'), 500
    except Exception:
        app.logger.exception('Inference failed')
        return jsonify(error='Inference failed; inspect local GPU log'), 500
    finally:
        if clothing_mask is not None:clothing_mask.end_request()
        lock.release()


if __name__ == '__main__':
    if os.name == 'nt':
        import msvcrt
        service_lock = (ROOT / 'instance/gpu-service.lock').open('a+b')
        if service_lock.seek(0,2) == 0:
            service_lock.write(b'0')
            service_lock.flush()
        service_lock.seek(0)
        try:
            msvcrt.locking(service_lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise SystemExit('The AI-FIT GPU service is already running or loading.')
    initialize()
    app.run(host='127.0.0.1', port=8001, threaded=True)
