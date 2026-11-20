"""Preprocessing regression checks; no weights or GPU required."""
import sys
import types
import unittest
import numpy as np
from PIL import Image

# The Windows parser is supplied by the user's existing installation.
sys.modules.setdefault('model', types.ModuleType('model'))
fake_schp = types.ModuleType('model.SCHP')
fake_schp.SCHP = object
sys.modules['model.SCHP'] = fake_schp
from clothing_mask import ClothingMask, restore_crop


class ClothingMaskTests(unittest.TestCase):
    def test_short_source_sampling_guard_excludes_bare_and_covered_arms(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.3,.2),(12,.7,.2),(13,.25,.5),(14,.75,.5)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        garment=Image.new('RGB',(100,180),'green');garment.info['long_sleeves']=True
        labels=np.full((400,300),5,np.uint8)
        with patch.object(self.masker,'_labels',return_value=labels):
            self.assertFalse(self.masker.short_source_upper(self.person,points,garment))
            labels[130:]=14
            self.assertTrue(self.masker.short_source_upper(self.person,points,garment))
            labels[:]=14
            self.assertFalse(self.masker.short_source_upper(self.person,points,garment))
            labels[:130]=5;garment.info['long_sleeves']=False
            self.assertFalse(self.masker.short_source_upper(self.person,points,garment))

    def test_request_cache_reuses_equal_pixels_but_not_mutation_or_later_requests(self):
        from unittest.mock import Mock
        masker=ClothingMask.__new__(ClothingMask)
        masker.parser=Mock(side_effect=lambda image:Image.new('L',image.size,5))
        photo=Image.new('RGB',(30,40),'blue')
        masker.begin_request()
        first=masker._labels(photo)
        np.testing.assert_array_equal(first,masker._labels(photo.copy()))
        self.assertEqual(masker.parser.call_count,1)
        photo.putpixel((0,0),(255,0,0));masker._labels(photo)
        self.assertEqual(masker.parser.call_count,2)
        for i in range(12):
            photo.putpixel((0,0),(i,0,0));masker._labels(photo)
        self.assertLessEqual(len(masker._request_labels),8)
        masker.end_request();self.assertIsNone(masker._request_labels)
        masker.begin_request();masker._labels(photo)
        self.assertEqual(masker.label_stats['calls'],1)
        self.assertEqual(masker.label_stats['hits'],0)

    def test_plain_long_reference_requires_clothing_at_both_source_elbows(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.3,.2),(12,.7,.2),(13,.25,.5),(14,.75,.5)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        garment=Image.new('RGB',(100,180),(35,65,40))
        garment.info.update(long_sleeves=True,garment_silhouette=Image.new('L',garment.size,255))
        labels=np.full((400,300),5,np.uint8)
        with patch.object(self.masker,'_labels',return_value=labels):
            self.assertTrue(self.masker.prefer_preserved_upper(self.person,points,garment))
            labels[130:,:]=14
            self.assertFalse(self.masker.prefer_preserved_upper(self.person,points,garment))
            labels[:]=5;garment.paste('white',(20,40,80,100))
            self.assertFalse(self.masker.prefer_preserved_upper(self.person,points,garment))
            garment.paste((35,65,40),(0,0,100,180));garment.info['long_sleeves']=False
            self.assertFalse(self.masker.prefer_preserved_upper(self.person,points,garment))

    def test_bare_collar_has_room_without_editing_upper_neck_or_face(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((0,150,55),(9,143,70),(10,157,70),(11,90,160),(12,210,160),(23,100,300),(24,200,300)):
            points[i]=SimpleNamespace(x=x/300,y=y/400,visibility=1.,presence=1.)
        self.labels[:]=0;self.labels[20:80,125:175]=13;self.labels[300:,90:210]=9
        fg=np.zeros((400,300),np.float32);fg[20:380,80:220]=1
        person,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),landmarks=points,
            foreground=Image.fromarray(fg),include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((118,125)),(0,0,255))
        for xy in ((150,65),(150,95),(150,350)):
            self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))

    def test_thigh_length_is_valid_but_long_sleeves_cannot_fake_a_long_body(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=1.,presence=1.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        edit=Image.new('L',self.person.size,255)
        labels=np.full((400,300),9,np.uint8);labels[:300]=5
        with patch.object(self.masker,'_labels',return_value=labels):
            self.assertTrue(self.masker.check_upper_length(self.person,points,'long',edit)['passed'])
        labels[:]=9;labels[:,20:65]=5;labels[:,235:280]=5
        with patch.object(self.masker,'_labels',return_value=labels):
            self.assertFalse(self.masker.check_upper_length(self.person,points,'long',edit)['passed'])

    def test_single_parser_mistake_cannot_erase_light_patterned_long_cloth(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=1.,presence=1.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        preview=Image.new('RGB',self.person.size,(240,225,180));edit=Image.new('L',self.person.size,255)
        with patch.object(self.masker,'_labels',return_value=np.full((400,300),9,np.uint8)):
            fixed=self.masker.restore_lower_layers(self.person,preview,points,edit,confirmed_lower=np.zeros((400,300),bool))
            self.assertEqual(fixed.getpixel((150,320)),preview.getpixel((150,320)))
            self.assertTrue(self.masker.check_upper_length(preview,points,'long',edit,independent_clothing=np.ones((400,300),bool))['passed'])

    def test_lighter_same_hue_fabric_is_not_an_invented_neutral_layer(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.1,visibility=1.,presence=1.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        garment=Image.new('RGB',(100,180),(60,90,120));garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        edit=Image.new('L',self.person.size,255)
        preview=Image.new('RGB',self.person.size,(145,170,195))
        self.assertTrue(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])
        preview.paste((200,200,200),(125,155,175,260))
        self.assertFalse(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])

    def test_long_upper_restores_generated_pants_below_real_garment_hem(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=1.,presence=1.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        labels=np.full((400,300),5,np.uint8);labels[330:]=9
        preview=Image.new('RGB',self.person.size,'blue');edit=Image.new('L',self.person.size,255)
        with patch.object(self.masker,'_labels',return_value=labels):
            result=self.masker.restore_lower_layers(self.person,preview,points,edit,confirmed_lower=labels==9)
        self.assertEqual(result.getpixel((150,300)),(0,0,255))
        self.assertEqual(result.getpixel((150,350)),self.person.getpixel((150,350)))

    def test_declared_long_top_rejects_a_waist_length_generation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        edit=Image.new('L',self.person.size);edit.paste(255,(90,260,210,360))
        self.assertFalse(self.masker.check_upper_length(self.person,points,'long',edit)['passed'])
        with patch.object(self.masker,'_labels',return_value=np.full((400,300),5,np.uint8)):
            self.assertTrue(self.masker.check_upper_length(self.person,points,'long',edit)['passed'])
        self.assertFalse(self.masker.check_upper_length(self.person,points,'standard',edit)['available'])

    def test_long_upper_can_cover_upper_trousers_but_preserves_lower_legs_face_and_scene(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((0,.5,.15),(11,.3,.25),(12,.7,.25),(23,.35,.65),(24,.65,.65),(25,.4,.90),(26,.6,.90)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        fg=Image.fromarray((self.labels>0).astype(np.float32))
        for length,expected in (('standard',self.person.getpixel((150,300))),('long',(0,0,255))):
            person,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),landmarks=points,
                include_protection=True,foreground=fg,upper_length=length)
            result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
            self.assertEqual(result.getpixel((150,300)),expected)
            for xy in ((150,380),(150,60),(15,300)):
                self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))

    def test_light_coat_mislabelled_at_brown_hair_edge_is_editable_but_blonde_core_stays(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        self.labels[80:220,110:140]=2
        for colour,expected in (((90,50,20),(0,0,255)),((230,210,170),(220,210,195))):
            source=self.person.copy();source.paste(colour,(110,80,140,220));source.paste((220,210,195),(110,80,115,220))
            person,mask,box,protected=self.masker.prepare(source,'upper',(384,512),landmarks=points,include_protection=True)
            result=restore_crop(Image.new('RGB',mask.size,'blue'),source,mask,box,protected)
            self.assertEqual(result.getpixel((113,170)),expected)
            self.assertEqual(result.getpixel((125,170)),colour)
            self.assertEqual(result.getpixel((113,90)),(220,210,195))

    def test_editable_long_sleeve_does_not_restore_misclassified_source_arm(self):
        from unittest.mock import patch
        source=Image.new('RGB',(50,50),(200,140,100));preview=Image.new('RGB',source.size,(25,60,40))
        labels=np.full((50,50),14,np.uint8)
        garment=Image.new('RGB',(100,200),'white');garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        edit=Image.new('L',source.size);edit.paste(255,(0,0,25,50))
        with patch.object(self.masker,'_labels',return_value=labels):
            fixed=self.masker.restore_exposed_skin(source,preview,edit_mask=edit,garment=garment)
        self.assertEqual(fixed.getpixel((12,25)),preview.getpixel((12,25)))
        self.assertEqual(fixed.getpixel((38,25)),source.getpixel((38,25)))

    def test_layered_source_requires_both_coat_and_inner_shirt(self):
        self.assertFalse(self.masker.has_layered_upper(self.person))
        self.labels[140:220,120:180]=7
        self.assertTrue(self.masker.has_layered_upper(self.person))

    def test_long_hair_core_and_head_are_preserved_while_old_shirt_fringe_can_change(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.1,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        self.labels[80:220,110:140]=2
        person,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),landmarks=points,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((125,170)),self.person.getpixel((125,170)))
        self.assertEqual(result.getpixel((125,90)),self.person.getpixel((125,90)))
        self.assertEqual(result.getpixel((110,170)),(0,0,255))

    def test_cropped_long_sleeves_cover_observed_arm_and_are_validated_without_wrist_guess(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        points=[SimpleNamespace(x=.5,y=.1,visibility=1.,presence=1.) for _ in range(33)]
        for i,x,y in ((11,.3,.25),(12,.7,.25),(23,.4,1.2),(24,.6,1.2),(15,.2,1.3),(16,.8,1.3)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=0.01 if i in (15,16,23,24) else 1.,presence=1.)
        self.labels[150:,50:80]=14
        garment=Image.new('RGB',(100,200),'white');garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        person,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),landmarks=points,
            include_protection=True,foreground=Image.new('F',self.person.size,1.),garment=garment)
        restored=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(restored.getpixel((65,350)),(0,0,255))
        edit=Image.new('L',self.person.size,255)
        self.assertFalse(self.masker.check_sleeve_coverage(self.person,points,garment,source=self.person,edit_mask=edit)['passed'])
        fabric=self.labels.copy();fabric[150:,50:80]=5
        with patch.object(self.masker,'_labels',side_effect=[fabric,self.labels]):
            self.assertTrue(self.masker.check_sleeve_coverage(self.person,points,garment,source=self.person,edit_mask=edit)['passed'])

    def test_donor_bare_arm_overrides_wide_short_shirt_silhouette_but_hands_do_not(self):
        from clothing_mask import long_sleeve_target
        self.labels[130:300,30:55]=14
        result=self.masker.garment_only(self.person,'upper')
        self.assertIs(long_sleeve_target(result),False)
        self.labels[130:300,30:55]=0
        self.labels[270:290,30:55]=14
        result=self.masker.garment_only(self.person,'upper')
        self.assertNotIn('long_sleeves',result.info)

    def test_dark_closed_reference_rejects_an_invented_light_inner_layer(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,90,90),(12,210,90),(23,100,290),(24,200,290)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        garment=Image.new('RGB',(100,160),(20,70,40))
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        preview=Image.new('RGB',self.person.size,(20,70,40))
        edit=Image.new('L',self.person.size,255)
        self.assertTrue(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])
        preview.paste('white',(140,170,160,280))
        self.assertFalse(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])
        preview.paste((145,145,145),(140,170,160,280))
        self.assertFalse(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])
        edit.paste(0,(140,170,160,280))
        self.assertTrue(self.masker.check_upper_surface(preview,points,garment,edit)['passed'])
        garment.paste('white',(30,40,70,140))
        self.assertFalse(self.masker.check_upper_surface(preview,points,garment,edit)['available'])

    def test_short_target_rejects_long_forearm_fabric(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((13,230,190),(15,235,290)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        garment=Image.new('RGB',(200,240),'white')
        alpha=Image.new('L',garment.size);alpha.paste(255,(40,10,160,230));alpha.paste(255,(0,20,200,80))
        garment.info['garment_silhouette']=alpha
        self.labels[190:300,220:250]=5
        report=self.masker.check_sleeve_coverage(self.person,points,garment)
        self.assertFalse(report['passed']);self.assertEqual(report['long_arms'],[15])
        self.labels[190:300,220:250]=14
        self.assertTrue(self.masker.check_sleeve_coverage(self.person,points,garment)['passed'])

    def test_plain_white_correction_removes_invented_emblem_but_keeps_reference_print(self):
        from unittest.mock import patch
        garment=Image.new('RGB',(100,100),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        preview=Image.new('RGB',(100,100),(225,225,225));preview.paste((20,20,20),(55,35,61,41))
        with patch.object(self.masker,'_labels',return_value=np.full((100,100),5,np.uint8)):
            fixed=self.masker.restore_plain_surface(preview,garment)
            self.assertGreater(min(fixed.getpixel((58,38))),180)
            garment.paste((20,20,20),(55,35,61,41))
            self.assertIs(self.masker.restore_plain_surface(preview,garment),preview)

    def test_long_target_rejects_exposed_forearm_and_accepts_fabric(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.5,visibility=0.,presence=0.) for _ in range(33)]
        points[13]=SimpleNamespace(x=.77,y=.5,visibility=1.,presence=1.)
        points[15]=SimpleNamespace(x=.78,y=.72,visibility=1.,presence=1.)
        garment=Image.new('RGB',(100,200),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        self.labels[190:300,220:250]=14
        report=self.masker.check_sleeve_coverage(self.person,points,garment)
        self.assertTrue(report['available'])
        self.assertFalse(report['passed'])
        self.labels[190:300,220:250]=7
        self.assertTrue(self.masker.check_sleeve_coverage(self.person,points,garment)['passed'])

    def test_plain_dark_emblem_repair_preserves_zip_print_and_protected_pixels(self):
        from unittest.mock import patch
        garment=Image.new('RGB',(100,100),(25,25,25))
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        garment.paste('white',(49,10,51,90))
        preview=garment.copy();preview.paste('white',(70,35,76,41))
        edit=Image.new('L',preview.size,255)
        with patch.object(self.masker,'_labels',return_value=np.full((100,100),5,np.uint8)):
            fixed=self.masker.restore_plain_surface(preview,garment,edit_mask=edit)
            self.assertLess(max(fixed.getpixel((73,38))),80)
            self.assertEqual(fixed.getpixel((50,38)),(255,255,255))
            edit.paste(0,(65,30,80,45))
            self.assertEqual(self.masker.restore_plain_surface(preview,garment,edit_mask=edit).getpixel((73,38)),(255,255,255))
            garment.paste('white',(70,35,76,41))
            self.assertIs(self.masker.restore_plain_surface(preview,garment),preview)

    def test_plain_reference_antialias_edge_is_not_mistaken_for_a_print(self):
        from unittest.mock import patch
        garment=Image.new('RGB',(100,100),'white');garment.paste((25,25,25),(11,11,89,89))
        alpha=Image.new('L',garment.size);alpha.paste(255,(10,10,90,90))
        garment.info['garment_silhouette']=alpha
        preview=Image.new('RGB',(100,100),(25,25,25));preview.paste('white',(70,35,76,41))
        with patch.object(self.masker,'_labels',return_value=np.full((100,100),5,np.uint8)):
            self.assertLess(max(self.masker.restore_plain_surface(preview,garment).getpixel((73,38))),80)

    def test_short_sleeve_target_can_leave_forearms_exposed(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        garment=Image.new('RGB',(200,240),'white')
        silhouette=Image.new('L',garment.size)
        silhouette.paste(255,(40,10,160,230));silhouette.paste(255,(0,20,200,80))
        garment.info['garment_silhouette']=silhouette
        self.assertTrue(self.masker.check_sleeve_coverage(self.person,points,garment)['passed'])

    def test_restore_exposed_skin_keeps_new_sleeves(self):
        from unittest.mock import patch
        original = Image.new('RGB',(30,30),(204,138,110))
        preview = Image.new('RGB',(30,30),(44,77,65))
        source = np.full((30,30),14,dtype=np.uint8)
        output = source.copy()
        output[:,:15] = 5
        with patch.object(self.masker,'_labels',side_effect=[source,output]):
            result = self.masker.restore_exposed_skin(original,preview)
        self.assertEqual(result.getpixel((22,15)),original.getpixel((22,15)))
        self.assertEqual(result.getpixel((8,15)),preview.getpixel((8,15)))

    def test_skin_restoration_does_not_reintroduce_old_cuff(self):
        from unittest.mock import patch
        source=Image.new('RGB',(100,100),(20,25,35))
        preview=Image.new('RGB',(100,100),'white')
        original_labels=np.zeros((100,100),dtype=np.uint8)
        original_labels[10:50,20:80]=5
        original_labels[50:90,30:70]=14
        preview_labels=np.zeros_like(original_labels)
        preview_labels[40:95,20:80]=14
        with patch.object(self.masker,'_labels',side_effect=[original_labels,preview_labels]):
            result=self.masker.restore_exposed_skin(source,preview)
        self.assertEqual(result.getpixel((50,50)),(255,255,255))
        self.assertEqual(result.getpixel((50,75)),source.getpixel((50,75)))

    def setUp(self):
        self.labels = np.zeros((400, 300), dtype=np.uint8)
        self.labels[40:90, 120:180] = 13
        self.labels[100:260, 80:220] = 5
        self.labels[260:390, 90:210] = 9
        self.person = Image.new('RGB', (300, 400), (200, 80, 40))
        self.masker = ClothingMask.__new__(ClothingMask)
        self.masker.parser = lambda image: Image.fromarray(self.labels)

    def test_upper_crop_preserves_face_pants_and_background(self):
        person, mask, box = self.masker.prepare(self.person, 'upper')
        self.assertEqual(person.size, (768, 1024))
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.size, self.person.size)
        for xy in ((150, 60), (150, 320), (20, 200)):
            self.assertEqual(result.getpixel(xy), self.person.getpixel(xy))
        self.assertEqual(result.getpixel((150, 180)), (0, 0, 255))

    def test_short_shirt_does_not_recolour_the_bottom_of_a_long_dress(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,90,100),(12,210,100),(23,100,250),(24,200,250)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[100:380,80:220]=6
        garment=Image.new('RGB',(200,180),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,180)),(0,0,255))
        self.assertEqual(result.getpixel((150,320)),self.person.getpixel((150,320)))
        garment.info['garment_silhouette']=Image.new('L',(100,300),255)
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,320)),(0,0,255))

    def test_short_body_hem_limit_does_not_cut_long_sleeves_below_hips(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,100),(12,90,100),(23,200,230),(24,100,230),
                      (13,230,190),(15,235,290)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[100:380,80:220]=6
        self.labels[110:295,221:248]=14
        garment=Image.new('RGB',(200,180),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,270)),(0,0,255))
        self.assertEqual(result.getpixel((150,320)),self.person.getpixel((150,320)))

    def test_background_cleanup_keeps_new_fabric_and_restores_scene(self):
        from unittest.mock import patch
        original=Image.new('RGB',(100,100),(90,130,80))
        original.paste((180,180,190),(30,20,70,90))
        preview=Image.new('RGB',original.size,(150,160,160))
        preview.paste((10,65,35),(40,25,60,90))
        labels=np.zeros((100,100),np.uint8);labels[25:90,40:60]=5
        foreground=Image.new('F',original.size,0);foreground.paste(1,(30,20,70,90))
        with patch.object(self.masker,'_labels',return_value=labels):
            result=self.masker.restore_background_edges(original,preview,foreground,Image.new('L',original.size,255))
        self.assertEqual(result.getpixel((10,30)),original.getpixel((10,30)))
        self.assertEqual(result.getpixel((50,50)),preview.getpixel((50,50)))
        self.assertNotEqual(result.getpixel((32,50)),preview.getpixel((32,50)))
        with patch.object(self.masker,'_labels',return_value=np.zeros_like(labels)):
            result=self.masker.restore_background_edges(original,preview,foreground,
                Image.new('L',original.size,255),preview_foreground=Image.new('F',original.size,1))
        self.assertEqual(result.getpixel((50,50)),preview.getpixel((50,50)))

    def test_helmet_mouth_uses_visible_nose_for_neck_anchor(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.16,visibility=1.,presence=1.) for _ in range(33)]
        points[9].y=points[10].y=.25
        self.assertEqual(self.masker.neck_anchor(self.person,points),'nose')
        points[9].y=points[10].y=.18
        self.assertEqual(self.masker.neck_anchor(self.person,points),'mouth')

    def test_sleeve_refinement_preserves_face_torso_and_hands(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((0,150,60),(11,210,110),(12,90,110),(13,230,200),
                      (15,235,288),(17,237,303),(19,240,305),(21,232,302),(23,200,260),(24,100,260)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        protected=Image.new('L',self.person.size)
        protected.paste(255,(120,40,180,90))
        repair=self.masker.prepare_sleeve_refinement(self.person,points,[15],(384,512),
            Image.new('L',self.person.size,255),protected)
        self.assertIsNotNone(repair)
        person,mask,box=repair
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,250)),(0,0,255))
        for xy in ((150,60),(150,180),(237,303)):
            self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))

    def test_wrist_guard_preserves_hand_but_replaces_old_fabric_cuff(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,230,200),(15,235,288),
                      (17,237,302),(19,239,306),(21,232,303)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[120:290,220:250]=5
        self.labels[290:315,225:248]=14
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((235,285)),(0,0,255))
        self.assertEqual(result.getpixel((237,302)),self.person.getpixel((237,302)))

    def test_upper_preserves_background_next_to_shoulder(self):
        foreground = Image.fromarray((self.labels>0).astype(np.float32))
        _,mask,box,protected = self.masker.prepare(
            self.person,'upper',(384,512),foreground=foreground,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((75,120)),self.person.getpixel((75,120)))
        self.assertEqual(result.getpixel((90,120)),(0,0,255))

    def test_empty_clothing_does_not_fall_back_to_large_rectangle(self):
        self.labels[:] = 0
        with self.assertRaises(ValueError):
            self.masker.prepare(self.person, 'upper')

    def test_shirtless_torso_requires_pose_skin_and_foreground_agreement(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,90,100),(12,210,100),(23,100,260),(24,200,260)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[100:260,80:220]=14
        foreground=Image.fromarray((self.labels>0).astype(np.float32))
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,foreground=foreground,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,180)),(0,0,255))
        for xy in ((150,60),(150,88),(150,320),(20,180)):
            self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))
        with self.assertRaises(ValueError):
            self.masker.prepare(self.person,'upper',landmarks=points)
        with self.assertRaises(ValueError):
            self.masker.prepare(self.person,'upper',landmarks=points,
                foreground=Image.new('F',self.person.size,0))

    def test_shirtless_forearms_mislabelled_as_background_can_get_full_sleeves(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,100),(12,90,100),(23,200,260),(24,100,260),
                      (13,230,190),(15,235,290)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[100:260,80:220]=0
        foreground=(self.labels>0).astype(np.float32)
        foreground[100:260,80:220]=1
        foreground[100:292,220:245]=1
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,foreground=Image.fromarray(foreground),include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,250)),(0,0,255))
        self.assertEqual(result.getpixel((20,250)),self.person.getpixel((20,250)))

    def test_shirtless_source_uses_focused_crop_even_with_visible_ankles(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((0,150,60),(11,90,100),(12,210,100),(23,100,260),
                      (24,200,260),(27,120,380),(28,180,380)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[100:260,80:220]=14
        foreground=Image.fromarray((self.labels>0).astype(np.float32))
        _,_,box=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,foreground=foreground)
        self.assertLess(box[2]-box[0],self.person.width)
        self.assertLess(box[3]-box[1],self.person.height)

    def test_parser_receives_bgr(self):
        seen = []
        self.masker.parser = lambda image: (seen.append(image.getpixel((0, 0))) or Image.fromarray(self.labels))
        self.masker.prepare(self.person, 'upper')
        self.assertEqual(seen, [(40, 80, 200)])

    def test_donor_removal_retains_short_sleeve_silhouette(self):
        from clothing_mask import long_sleeve_target
        self.labels[:]=0
        self.labels[30:80,130:170]=13
        self.labels[110:300,105:195]=5
        self.labels[110:165,65:235]=5
        cleaned=self.masker.garment_only(self.person,'upper')
        self.assertIn('garment_silhouette',cleaned.info)
        self.assertFalse(long_sleeve_target(cleaned))

    def test_flat_hoodie_is_not_cut_when_pose_hallucinates_a_person(self):
        self.labels[:]=5
        # A small patch at a hood opening is sometimes parsed as face.
        self.labels[30:50,140:175]=13
        garment=self.person.copy()
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        self.assertIs(self.masker.garment_only(garment,'upper'),garment)

    def test_split_hem_can_close_without_replacing_lower_trousers(self):
        self.labels[245:260, 140:160] = 9
        person, mask, box = self.masker.prepare(self.person, 'upper')
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.getpixel((150, 252)), (0, 0, 255))
        self.assertEqual(result.getpixel((150, 320)), self.person.getpixel((150, 320)))

    def test_prepare_uses_requested_size(self):
        person, mask, box = self.masker.prepare(self.person, 'upper', (384, 512))
        self.assertEqual(person.size, (384, 512))
        self.assertEqual(mask.size, (384, 512))

    def test_short_target_sleeves_preserve_exposed_forearms(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,230,200),(15,235,288)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[124:290,221:245]=14
        garment=Image.new('RGB',(200,240),'white')
        silhouette=Image.new('L',garment.size)
        silhouette.paste(255,(40,10,160,230))
        silhouette.paste(255,(0,20,200,80))
        garment.info['garment_silhouette']=silhouette
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,255)),self.person.getpixel((233,255)))
        self.assertEqual(result.getpixel((150,180)),(0,0,255))

    def test_short_target_removes_source_cuff_mislabelled_as_skin(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,235,215),(15,237,288)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[120:194,215:250]=5
        self.labels[194:290,220:247]=14
        garment=Image.new('RGB',(200,240),'white')
        silhouette=Image.new('L',garment.size)
        silhouette.paste(255,(40,10,160,230));silhouette.paste(255,(0,20,200,80))
        garment.info['garment_silhouette']=silhouette
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'white'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((234,196)),(255,255,255))
        self.assertEqual(result.getpixel((234,255)),self.person.getpixel((234,255)))

    def test_short_bottoms_preserve_existing_bare_legs(self):
        self.labels[300:355,110:190]=16
        garment=Image.new('RGB',(200,140),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        _,mask,box,protected=self.masker.prepare(self.person,'lower',(384,512),
            garment=garment,include_protection=True)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,320)),self.person.getpixel((150,320)))
        self.assertEqual(result.getpixel((150,280)),(0,0,255))

    def test_long_sleeved_target_keeps_full_body_context_when_ankles_are_visible(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.5,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((0,.5,.15),(11,.7,.27),(12,.3,.27),(13,.77,.5),
                      (14,.23,.5),(15,.78,.7),(16,.22,.7),(23,.6,.65),(24,.4,.65),
                      (27,.6,.94),(28,.4,.94)):
            points[i]=SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        garment=Image.new('RGB',(200,240),'white')
        garment.info['garment_silhouette']=Image.new('L',garment.size,255)
        _,_,box=self.masker.prepare(self.person,'upper',(384,512),landmarks=points,garment=garment)
        self.assertLessEqual(box[0],0)
        self.assertLessEqual(box[1],0)
        self.assertGreaterEqual(box[2],self.person.width)
        self.assertGreaterEqual(box[3],self.person.height)

    def test_wide_photo_crop_contains_all_editable_pixels(self):
        labels=np.zeros((160,600),dtype=np.uint8)
        labels[30:120,40:560]=5
        self.masker.parser=lambda image:Image.fromarray(labels)
        image=Image.new('RGB',(600,160),'red')
        _,mask,box,protected=self.masker.prepare(image,'upper',(384,512),include_protection=True)
        self.assertLess(box[1],0)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),image,mask,box,protected)
        for point in ((45,70),(555,70),(300,70)):
            self.assertEqual(result.getpixel(point),(0,0,255))

    def test_hands_near_shirt_are_preserved(self):
        self.labels[215:270, 215:235] = 14
        person, mask, box = self.masker.prepare(self.person, 'upper')
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.getpixel((220, 245)), self.person.getpixel((220, 245)))

    def test_upper_sleeve_can_reach_forearm_without_editing_hand_or_background(self):
        from types import SimpleNamespace
        points = [SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,230,200),(15,235,288),
                      (17,237,307),(19,240,304)):
            points[i] = SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[124:310,221:245] = 14
        _,mask,box,protected = self.masker.prepare(
            self.person,'upper',(384,512),landmarks=points,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,255)),(0,0,255))
        for point in ((238,301),(270,255),(150,320),(150,60)):
            self.assertEqual(result.getpixel(point),self.person.getpixel(point))

    def test_long_sleeves_have_fabric_space_beyond_bare_forearm(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,230,200),(15,235,288),(17,237,307),(19,240,304)):
            points[i]=SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[124:310,221:245]=14
        foreground=Image.fromarray((self.labels>0).astype(np.float32))
        _,mask,box,protected=self.masker.prepare(self.person,'upper',(384,512),
            landmarks=points,include_protection=True,foreground=foreground)
        result=restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((251,255)),(0,0,255))
        self.assertEqual(result.getpixel((275,255)),self.person.getpixel((275,255)))
        self.assertEqual(result.getpixel((238,301)),self.person.getpixel((238,301)))

    def test_restore_does_not_blend_original_garment_colour(self):
        _, mask, box = self.masker.prepare(self.person, 'upper')
        faded = Image.new('L', mask.size, 80)
        faded.paste(mask.point(lambda v: 80 if v else 0))
        result = restore_crop(Image.new('RGB', mask.size, 'blue'), self.person, faded, box)
        self.assertEqual(result.getpixel((150, 180)), self.person.getpixel((150, 180)))


    def test_overall_skin_and_background_are_not_regenerated(self):
        self.labels[190:225,120:180]=14
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        for xy in ((150,205),(78,200),(150,60)):
            self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))
        self.assertEqual(result.getpixel((150,300)),(0,0,255))

    def test_full_resolution_protection_survives_small_mask(self):
        self.labels[200,150]=14
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(24,32),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,200)),self.person.getpixel((150,200)))

    def test_donor_extraction_removes_head_skin_and_background(self):
        self.labels[200:220,120:180]=14
        result=self.masker.garment_only(self.person,'overall')
        colors=set(map(tuple,np.asarray(result).reshape(-1,3)))
        self.assertEqual(colors,{(255,255,255),(200,80,40)})
        self.assertLess(result.height,self.person.height)
        self.labels[:]=0
        with self.assertRaises(ValueError):self.masker.garment_only(self.person,'overall')


    def test_leg_repair_requires_both_foreground_and_pose(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        points[11].x=.3;points[12].x=.7
        for i in (25,26):points[i].y=.7
        for i in (27,28):points[i].y=.9
        self.labels[280:360,140:160]=0
        for probability,expected in ((0.,self.person.getpixel((150,320))),(1.,(0,0,255))):
            foreground=Image.fromarray(np.full((400,300),probability,dtype=np.float32))
            person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),landmarks=points,include_protection=True,foreground=foreground)
            result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
            self.assertEqual(result.getpixel((150,320)),expected)
            self.assertEqual(result.getpixel((20,320)),self.person.getpixel((20,320)))


    def test_overall_replaces_leg_labelled_fabric_but_preserves_shoes(self):
        self.labels[280:340,120:180]=16
        self.labels[360:390,120:180]=18
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,310)),(0,0,255))
        self.assertEqual(result.getpixel((150,375)),self.person.getpixel((150,375)))

    def test_overall_edge_repair_needs_person_foreground(self):
        for probability,expected in ((0.,self.person.getpixel((77,160))),(1.,(0,0,255))):
            foreground=Image.fromarray(np.full((400,300),probability,dtype=np.float32))
            person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True,foreground=foreground)
            result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
            self.assertEqual(result.getpixel((77,160)),expected)
            self.assertEqual(result.getpixel((20,160)),self.person.getpixel((20,160)))

    def test_overall_sleeve_and_cuff_mask_preserves_forearm_and_shoe(self):
        from types import SimpleNamespace
        points = [SimpleNamespace(x=.5,y=.5,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.27,.26),(12,.73,.26),(13,.27,.47)):
            points[i] = SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        self.labels[104:208,75:90] = 14
        self.labels[360:390,120:180] = 18
        _,mask,box,protected = self.masker.prepare(
            self.person,'overall',(384,512),landmarks=points,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((83,114)),(0,0,255))
        self.assertEqual(result.getpixel((83,187)),self.person.getpixel((83,187)))
        self.assertEqual(result.getpixel((150,356)),(0,0,255))
        self.assertEqual(result.getpixel((150,366)),self.person.getpixel((150,366)))


if __name__ == '__main__': unittest.main()
