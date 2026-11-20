import io,sys,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import local_windows as service
class LocalWindowsTests(unittest.TestCase):
    def test_progress_requires_service_auth_and_invalid_input_finishes_failed(self):
        from progress import Progress
        import io
        key='d'*32;client=service.app.test_client();photo=io.BytesIO();Image.new('RGB',(128,128)).save(photo,format='PNG');photo.seek(0)
        with patch.object(service,'progress',Progress()),patch.object(service,'pipeline',Mock()),patch.object(service,'preprocessor',Mock()),patch.object(service,'clothing_mask',Mock()):
            self.assertEqual(client.get('/progress/'+key).status_code,401)
            response=client.post('/infer',headers={'Authorization':'Bearer '+service.KEY,'X-AIFIT-Request-ID':key},data={'person':(photo,'person.png')})
            self.assertEqual(response.status_code,400)
            snapshot=client.get('/progress/'+key,headers={'Authorization':'Bearer '+service.KEY}).json
            self.assertEqual(snapshot['status'],'failed');self.assertLess(snapshot['percent'],100)

    def test_ram_cache_skips_parking_when_memory_reserve_is_unavailable(self):
        import psutil
        from types import SimpleNamespace
        renderer=Mock();renderer.park_bytes.return_value=2*1024**3
        with patch.object(psutil,'virtual_memory',return_value=SimpleNamespace(available=3*1024**3)):
            self.assertFalse(service.can_park_renderer(renderer))
        with patch.object(psutil,'virtual_memory',return_value=SimpleNamespace(available=4*1024**3)):
            self.assertTrue(service.can_park_renderer(renderer))

    def test_fast_profile_preserves_full_quality_retry_and_long_outfits(self):
        with patch.dict(service.os.environ,{'AIFIT_SPEED_PROFILE':'fast'}):
            self.assertEqual(service.generation_steps(True,'upper','standard',1),12)
            self.assertEqual(service.generation_steps(True,'upper','standard',1,True),15)
            self.assertEqual(service.generation_steps(False,'upper','standard',1),35)
            self.assertEqual(service.generation_steps(True,'upper','standard',2),30)
            self.assertEqual(service.generation_steps(False,'upper','standard',2),50)
            self.assertEqual(service.generation_steps(True,'upper','long',1),30)
            self.assertEqual(service.generation_steps(False,'upper','long',1),50)

    def test_warm_switch_keeps_only_one_cuda_renderer_and_reuses_cpu_weights(self):
        models=[];loads=[]
        class Renderer:
            def __init__(self,name):self.name=name;self.device='cuda';models.append(self)
            def park_cpu(self):self.device='cpu'
            def activate_cuda(self):
                self.assert_other_cpu()
                self.device='cuda'
            def assert_other_cpu(self):
                assert all(m is self or m.device=='cpu' for m in models)
        first=Renderer('fashn')
        def load_deco():
            loads.append('deco');first.assert_other_cpu()
            self.assertEqual(first.device,'cpu')
            service.pipeline=Renderer('deco');service.backend_name='deco';service.infer_size=(512,768)
        with patch.object(service,'pipeline',first),patch.object(service,'backend_name','fashn'),patch.object(service,'infer_size',(576,864)),patch.object(service,'cached_renderer',None),patch.object(service,'can_park_renderer',return_value=True),patch.object(service,'deco_available',return_value=True),patch.object(service,'_load_deco',side_effect=load_deco),patch.object(service.torch.cuda,'empty_cache'):
            service.activate_engine('deco')
            service.activate_engine('fashn')
            self.assertIs(service.pipeline,first)
            self.assertEqual(service.infer_size,(576,864))
            service.activate_engine('deco')
            self.assertEqual(loads,['deco'])
            self.assertEqual([m.device for m in models],['cpu','cuda'])
            self.assertEqual(service.cached_renderer[0],'fashn')

    def test_failed_warm_activation_restores_previous_renderer(self):
        previous=Mock();target=Mock();target.activate_cuda.side_effect=RuntimeError('simulated CUDA failure')
        with patch.object(service,'pipeline',previous),patch.object(service,'backend_name','fashn'),patch.object(service,'cached_renderer',('deco',target,(512,768))),patch.object(service,'can_park_renderer',return_value=True),patch.object(service,'deco_available',return_value=True),patch.object(service.torch.cuda,'empty_cache'):
            with self.assertRaisesRegex(RuntimeError,'Renderer activation failed'):
                service.activate_engine('deco')
            self.assertIs(service.pipeline,previous)
            self.assertEqual(service.backend_name,'fashn')
            previous.activate_cuda.assert_called_once()
            self.assertIsNone(service.cached_renderer)

    def test_fast_steps_keep_quality_retry_and_long_outfits(self):
        self.assertEqual(service.generation_steps(True,'upper','standard',1),20)
        self.assertEqual(service.generation_steps(True,'upper','standard',2),30)
        self.assertEqual(service.generation_steps(True,'upper','long',1),30)
        self.assertEqual(service.generation_steps(True,'lower','standard',1),30)
        self.assertEqual(service.generation_steps(False,'upper','standard',1),50)

    def test_adaptive_plain_upper_respects_explicit_engine_length_and_calibration(self):
        image=Image.new('RGB',(10,10));masker=Mock();masker.prefer_preserved_upper.return_value=True
        with patch.object(service,'clothing_mask',masker),patch.object(service,'deco_available',return_value=True),patch.object(service,'preferred_engine',side_effect=lambda image,default:default):
            self.assertEqual(service.image_engine(image,'upper','auto','standard',image,[],image),'deco')
            self.assertEqual(service.image_engine(image,'upper','fashn','standard',image,[],image),'fashn')
            self.assertEqual(service.image_engine(image,'upper','auto','long',image,[],image),'fashn')
            masker.prefer_preserved_upper.return_value=False
            self.assertEqual(service.image_engine(image,'upper','auto','standard',image,[],image),'fashn')
            masker.prefer_preserved_upper.return_value=True
            with patch.object(service,'preferred_engine',return_value='fashn'):
                self.assertEqual(service.image_engine(image,'upper','auto','standard',image,[],image),'fashn')

    def test_waiting_interactive_request_precedes_background_audit(self):
        import threading
        acquired=threading.Event();release=threading.Event()
        def shopper():
            if service.acquire_slot(timeout=2):
                acquired.set();release.wait(2);service.lock.release()
        service.lock.acquire();worker=threading.Thread(target=shopper)
        try:
            worker.start()
            with service.admission:
                self.assertTrue(service.admission.wait_for(lambda:service.foreground_waiting==1,timeout=1))
            self.assertFalse(service.acquire_slot(background=True))
            service.lock.release()
            self.assertTrue(acquired.wait(1))
            self.assertFalse(service.acquire_slot(background=True))
        finally:
            release.set();worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(service.foreground_waiting,0)
        self.assertFalse(service.lock.locked())

    def test_queue_timeout_and_queue_bound_do_not_leak_waiters(self):
        service.lock.acquire()
        try:
            self.assertFalse(service.acquire_slot(timeout=0))
            self.assertEqual(service.foreground_waiting,0)
            with patch.object(service,'foreground_waiting',2):
                self.assertFalse(service.acquire_slot(timeout=0))
        finally:service.lock.release()

    def setUp(self):
        # Route tests must not depend on installed model assets or host engine
        # settings. Individual selection tests override these defaults.
        selector=patch.object(service,'select_engine',return_value='fashn')
        selector.start();self.addCleanup(selector.stop)
        available=patch.object(service,'deco_available',return_value=False)
        available.start();self.addCleanup(available.stop)
        crop=patch.object(service,'cropped_upper',return_value=False)
        crop.start();self.addCleanup(crop.stop)
        environment=patch.dict(service.os.environ,{'AIFIT_ENGINE':'auto'})
        environment.start();self.addCleanup(environment.stop)

    def test_automatic_upper_repair_preserves_nonupper_renderer_and_explicit_choices(self):
        image=Image.new('RGB',(10,10))
        with patch.object(service,'deco_available',return_value=True),patch.object(service,'preferred_engine',side_effect=lambda image,default:default):
            self.assertEqual(service.requested_engine(image,'upper'),'fashn')
            for category in ('lower','overall'):
                self.assertEqual(service.requested_engine(image,category),'deco')
                self.assertEqual(service.requested_engine(image,category,'fashn'),'fashn')
            with patch.dict(service.os.environ,{'AIFIT_ENGINE':'fashn'}):
                self.assertEqual(service.requested_engine(image,'lower'),'fashn')

    def test_failed_fashn_sleeves_use_deco_without_relaxing_validation(self):
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        prep=Mock();prep.validate_person.return_value=([object()]*33,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
        prep.garment_has_person.return_value=False
        prep.align_upper_shoulders.side_effect=lambda points,output,**kw:output
        prep.check_output_pose.return_value={'passed':True}
        masker=Mock();masker.neck_anchor.return_value='mouth'
        mask=Image.new('L',(128,128),255);protected=Image.new('L',(128,128))
        masker.prepare.return_value=(Image.new('RGB',(128,128)),mask,(0,0,128,128),protected)
        masker.prepare_sleeve_refinement.return_value=(Image.new('RGB',(128,128)),mask,(0,0,128,128))
        masker.restore_exposed_skin.side_effect=lambda original,preview,**kwargs:preview
        masker.restore_plain_surface.side_effect=lambda preview,garment,**kwargs:preview
        masker.restore_background_edges.side_effect=lambda original,preview,*a,**kw:preview
        failed={'passed':False,'short_arms':[15]}
        masker.check_sleeve_coverage.side_effect=[failed,failed,{'passed':True}]
        masker.check_upper_surface.return_value={'passed':True}
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
        def switch(engine):service.backend_name=engine
        with patch.object(service,'backend_name','fashn'),patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service,'deco_available',return_value=True),patch.object(service,'activate_engine',side_effect=switch) as activation,patch.object(service.torch,'Generator',return_value=Mock()),patch.dict(service.os.environ,{'AIFIT_ENGINE':'auto'}):
            response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'upper'})
        self.assertEqual(response.status_code,200)
        self.assertIn(('deco',),[call.args for call in activation.call_args_list])
        self.assertEqual([call.kwargs['num_inference_steps'] for call in pipeline.call_args_list],[20,20,50])
        self.assertEqual(response.headers['X-AIFIT-Backend'],'deco')
        self.assertEqual(prep.check_output_pose.call_count,3)

    def test_engine_switch_discards_inactive_renderer_before_loading(self):
        import weakref
        class Old:pass
        original=service.pipeline
        service.pipeline=Old();reference=weakref.ref(service.pipeline)
        def load():
            self.assertIsNone(reference())
            service.pipeline=object();service.backend_name='fashn'
        try:
            with patch.object(service,'backend_name','deco'),patch.object(service,'_load_fashn',side_effect=load),patch.object(service.torch.cuda,'empty_cache'):
                service.activate_engine('fashn')
                self.assertEqual(service.backend_name,'fashn')
        finally:service.pipeline=original

    def test_reference_calibration_is_keyed_by_image_pixels(self):
        import tempfile,json
        from pathlib import Path
        image=Image.new('RGB',(10,10),'black')
        with tempfile.TemporaryDirectory() as directory,patch.object(service,'ROOT',Path(directory)):
            folder=Path(directory)/'instance';folder.mkdir()
            (folder/'garment-backends.json').write_text(json.dumps({'garments':{
                service.garment_signature(image):{'engine':'deco'}}}))
            self.assertEqual(service.preferred_engine(image,'fashn'),'deco')
            changed=image.copy();changed.putpixel((0,0),(255,255,255))
            self.assertEqual(service.preferred_engine(changed,'fashn'),'fashn')

    def test_fashn_retries_opened_garment_with_stronger_masked_conditioning(self):
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        prep=Mock();prep.validate_person.return_value=([object()]*33,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
        prep.garment_has_person.return_value=False
        prep.align_upper_shoulders.side_effect=lambda points,output,**kw:output
        prep.check_output_pose.return_value={'passed':True}
        masker=Mock();masker.neck_anchor.return_value='mouth'
        masker.prepare.return_value=(Image.new('RGB',(128,128)),Image.new('L',(128,128),255),
            (0,0,128,128),Image.new('L',(128,128)))
        masker.restore_exposed_skin.side_effect=lambda original,preview,**kwargs:preview
        masker.restore_plain_surface.side_effect=lambda preview,garment,**kwargs:preview
        masker.restore_background_edges.side_effect=lambda original,preview,*a,**kw:preview
        masker.check_sleeve_coverage.return_value={'passed':True}
        masker.check_upper_surface.side_effect=[{'passed':False},{'passed':True}]
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
        with patch.object(service,'backend_name','fashn'),patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service.torch,'Generator',return_value=Mock()):
            response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'upper'})
        self.assertEqual(response.status_code,200)
        first,second=pipeline.call_args_list
        self.assertEqual(first.kwargs['num_inference_steps'],20)
        self.assertTrue(first.kwargs['segmentation_free'])
        self.assertFalse(second.kwargs['segmentation_free'])
        self.assertEqual(second.kwargs['guidance_scale'],3.0)
        self.assertEqual(response.headers['X-AIFIT-Backend'],'fashn')

    def test_sleeve_refinement_returns_only_a_rechecked_passing_candidate(self):
        from preprocessing import InputRejected
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128),(50,100,150)).save(stream,format='PNG');stream.seek(0);return stream
        for final_pass in (True,False):
            prep=Mock();prep.validate_person.return_value=([object()]*33,None)
            prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
            prep.garment_has_person.return_value=False
            prep.align_upper_shoulders.side_effect=lambda points,output,**kw:output
            prep.check_output_pose.return_value={'passed':True}
            masker=Mock();masker.neck_anchor.return_value='mouth'
            mask=Image.new('L',(128,128),255);protected=Image.new('L',(128,128))
            masker.prepare.return_value=(Image.new('RGB',(128,128)),mask,(0,0,128,128),protected)
            masker.prepare_sleeve_refinement.return_value=(Image.new('RGB',(128,128)),mask,(0,0,128,128))
            masker.restore_exposed_skin.side_effect=lambda original,preview,**kwargs:preview
            masker.restore_plain_surface.side_effect=lambda preview,garment,**kwargs:preview
            masker.restore_background_edges.side_effect=lambda original,preview,*a,**kw:preview
            failed={'passed':False,'short_arms':[15]}
            masker.check_sleeve_coverage.side_effect=[failed,{'passed':True}] if final_pass else [failed,failed,failed]
            pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
            with patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service.torch,'Generator',return_value=Mock()):
                response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                    data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'upper'})
            self.assertEqual(response.status_code,200 if final_pass else 422)
            self.assertEqual(masker.prepare_sleeve_refinement.call_count,1)
            self.assertEqual(pipeline.call_count,2 if final_pass else 3)
            self.assertEqual(pipeline.call_args_list[1].kwargs['num_inference_steps'],35)
            if final_pass:self.assertEqual(response.headers['X-AIFIT-Sleeve-Refinement'],'1')

    def test_upper_alignment_reapplies_original_protected_pixels(self):
        def photo(color):
            stream=io.BytesIO();Image.new('RGB',(128,128),color).save(stream,format='PNG');stream.seek(0);return stream
        prep=Mock();prep.validate_person.return_value=([object()]*33,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
        prep.garment_has_person.return_value=False
        prep.align_upper_shoulders.return_value=Image.new('RGB',(128,128),'blue')
        prep.check_output_pose.return_value={'passed':True}
        protected=Image.new('L',(128,128));protected.paste(255,(0,0,40,40))
        masker=Mock();masker.prepare.return_value=(Image.new('RGB',(128,128)),
            Image.new('L',(128,128),255),(0,0,128,128),protected)
        masker.restore_exposed_skin.side_effect=lambda original,preview,**kwargs:preview
        masker.restore_plain_surface.side_effect=lambda preview,garment,**kwargs:preview
        masker.restore_background_edges.side_effect=lambda original,preview,*args,**kwargs:preview
        masker.neck_anchor.return_value='mouth'
        masker.check_sleeve_coverage.return_value={'passed':True}
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'red')]);pipeline.last_safety_blocked=False
        with patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service.torch,'Generator',return_value=Mock()):
            response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                data={'person':(photo((60,120,180)),'p.png'),'garment':(photo('white'),'g.png'),'category':'upper'})
        self.assertEqual(response.status_code,200)
        result=Image.open(io.BytesIO(response.data))
        for got,want in zip(result.getpixel((15,15)),(60,120,180)):
            self.assertAlmostEqual(got,want,delta=3)
        self.assertGreater(result.getpixel((90,90))[2],245)
        prep.align_upper_shoulders.assert_called_once()
        prep.check_output_pose.assert_called_once()

    def test_output_quality_retry_returns_only_a_passing_candidate(self):
        from preprocessing import InputRejected
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        for errors,expected_calls,expected_status in [([InputRejected('bad pose'),{}],2,200),
                                                     ([InputRejected('bad pose')]*2,2,422),
                                                     ([{}],1,200)]:
            prep=Mock();prep.validate_person.return_value=([object()]*33,None)
            prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
            prep.garment_has_person.return_value=False
            prep.check_output_pose.side_effect=errors
            masker=Mock();masker.prepare.return_value=(Image.new('RGB',(128,128)),Image.new('L',(128,128),255),(0,0,128,128),Image.new('L',(128,128)))
            pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
            with patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service.torch,'Generator',return_value=Mock()):
                response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                    data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'overall'})
            self.assertEqual(response.status_code,expected_status)
            self.assertEqual(pipeline.call_count,expected_calls)
            self.assertFalse(service.lock.locked())
            if expected_status==200:
                self.assertEqual(response.headers['X-AIFIT-Attempts'],str(expected_calls))

    def test_safety_rejection_is_not_retried(self):
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        prep=Mock();prep.validate_person.return_value=([object()]*33,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white');prep.garment_has_person.return_value=False
        masker=Mock();masker.prepare.return_value=(Image.new('RGB',(128,128)),Image.new('L',(128,128),255),(0,0,128,128),Image.new('L',(128,128)))
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=True
        with patch.object(service,'pipeline',pipeline),patch.object(service,'preprocessor',prep),patch.object(service,'clothing_mask',masker),patch.object(service.torch,'Generator',return_value=Mock()):
            response=service.app.test_client().post('/infer',headers={'Authorization':'Bearer '+service.KEY},
                data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'overall'})
        self.assertEqual(response.status_code,422)
        self.assertEqual(pipeline.call_count,1)
        prep.check_output_pose.assert_not_called()
        self.assertFalse(service.lock.locked())

    def test_health_without_cuda_is_503(self):
        with patch.object(service,'pipeline',None),patch.object(service.torch.cuda,'is_available',return_value=False):
            response=service.app.test_client().get('/health')
        self.assertEqual(response.status_code,503);self.assertFalse(response.json['ready']);self.assertIsNone(response.json['gpu'])
    def test_authorization_and_startup(self):
        client=service.app.test_client()
        self.assertEqual(client.post('/infer').status_code,401)
        with patch.object(service,'pipeline',None):
            self.assertEqual(client.post('/infer',headers={'Authorization':'Bearer '+service.KEY}).status_code,503)
    def test_route_preserves_garment_alpha(self):
        prep=Mock();prep.validate_person.return_value=([],None);prep.clean_garment.side_effect=ValueError('stop before inference')
        person=io.BytesIO();Image.new('RGB',(128,128)).save(person,format='PNG');person.seek(0)
        garment=io.BytesIO();Image.new('RGBA',(128,128),(20,30,40,0)).save(garment,format='PNG');garment.seek(0)
        with patch.object(service,'pipeline',Mock()),patch.object(service,'clothing_mask',Mock()),patch.object(service,'preprocessor',prep):
            response=service.app.test_client().post('/infer',data={'person':(person,'p.png'),'garment':(garment,'g.png')},headers={'Authorization':'Bearer '+service.KEY})
        self.assertEqual(response.status_code,422)
        im=prep.clean_garment.call_args.args[0];self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getpixel((0,0))[3],0)
        self.assertFalse(service.lock.locked())
    def test_route_uses_pose_and_rejects_changed_output(self):
        from preprocessing import InputRejected
        prep=Mock();points=[object()]*33;prep.validate_person.return_value=(points,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
        prep.garment_has_person.return_value=False
        prep.check_output_pose.side_effect=InputRejected('pose changed')
        masker=Mock();masker.prepare.return_value=(Image.new('RGB',(128,128)),Image.new('L',(128,128),255),(0,0,128,128),Image.new('L',(128,128)))
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        with patch.object(service,'pipeline',pipeline),patch.object(service,'clothing_mask',masker),patch.object(service,'preprocessor',prep):
            response=service.app.test_client().post('/infer',data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'overall'},headers={'Authorization':'Bearer '+service.KEY})
        self.assertEqual(response.status_code,422)
        self.assertIs(masker.prepare.call_args.kwargs['landmarks'],points)
        self.assertFalse(service.lock.locked())
    def test_cuda_runtime_oom_is_retryable_and_releases_lock(self):
        def image():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        for error,status in [(RuntimeError('CUDA error: out of memory'),503),(RuntimeError('tensor shape mismatch'),500)]:
            with self.subTest(status=status):
                prep=Mock();prep.validate_person.side_effect=error
                with patch.object(service,'pipeline',Mock()),patch.object(service,'clothing_mask',Mock()),patch.object(service,'preprocessor',prep),patch.object(service.torch.cuda,'empty_cache') as clear,patch.object(service.app.logger,'exception'):
                    response=service.app.test_client().post('/infer',data={'person':(image(),'p.png'),'garment':(image(),'g.png')},headers={'Authorization':'Bearer '+service.KEY})
                self.assertEqual(response.status_code,status)
                self.assertEqual(clear.call_count,1 if status==503 else 0)
                self.assertFalse(service.lock.locked())
                self.assertEqual(response.mimetype,'application/json')
if __name__=='__main__':unittest.main()
