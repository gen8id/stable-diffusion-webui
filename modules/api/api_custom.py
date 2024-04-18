# added by sungjoon.kim at 24'04.15

from fastapi.exceptions import HTTPException

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart, shared_items, script_callbacks, infotext_utils, sd_models
from modules.api import models
from modules.api.models import PydanticModelGenerator
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from contextlib import closing
from modules.progress import create_task_id, add_task_to_queue, start_task, finish_task, current_task

# added by sungjoon.kim at 24'04.15
import modules.api.api as api
from pydantic import BaseModel
import deepl


API_URL_DEEPL    = 'https://api-free.deepl.com'
API_KEY_DEEPL    = '9b534bc9-7729-44ae-b4f4-2fc0ba33968a:fx'
pstv_pmpt_prefix = "(realistic, photo realistic, dramatic lighting, full body, best quality:1.4), (side lighting, finely detailed beautiful eyes: 1.3), (raw photo, masterpiece, ultra detailed:1.2), (highres, aegyo sal, Polaroid dark tone low key film:1.1), extremely detailed CG unity 8k wallpaper, extremely delicate and beautiful, amazing, finely detail, official art, huge filesize, extremely detailed, extremely detailed eyes and face, light on face, subsurface scattering, amazing fine detail, Nikon D850 film stock photograph Kodak Portra 400 camera f1.6 lens, rich colors, lifelike texture, ultra high res, side look, professional lighting, hyper realism, sexy, beautiful, big eyes, beautiful detailed eyes, high quality makeup, cute, slim waist, 1{{gender}}, (mid 20's a little muscular kpop idol {{gender}}:1.1), (sharp:0.7)"
pstv_pmpt_suffix = ", <lora:kitagawa_marin_v1-1:1>, <lora:Asian orgasm v4:1.3>"
ngtv_pmpt        = "(worst quality, low quality, normal quality:1.8), (monochrome, grayscale:1.2), (sweat:1.1), lowres, paintings, sketches, nipples, skin spots, acnes, skin blemishes, bad anatomy, tilted head, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross, easynegative, negative_hand-neg, wearing vests, ng_deepnegative_v1_75t, 2girls, 2 girls, two girls, petals, logo, nametag, watermark, tattoo, tattoos, leather, cum, dripping, wet, tattoo, veins, FastNegativeV2, fat, v, peace symbol, nsfw,  dirty face, 1leg, 1arm, 1 leg, 1 arm, one leg, one arm, blurred eyes, too big breast, naked, bare chested, underwear, lingerie, too short pants, naked abdomen, naked belly, naked stomach, 2head, 2 head, two head, reflected light in eyes"


StableDiffusionTxt2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": str, "default": "DPM++ 2M"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "force_task_id", "type": str, "default": None},
        {"key": "infotext", "type": str, "default": None},
        {"key": "gndr", "type": str, "default": None},
        {"key": "cstm", "type": str, "default": None},
        {"key": "look", "type": str, "default": None},
        {"key": "bgnd", "type": str, "default": None}
    ]
).generate_model()


class Txt2ImgParam(BaseModel):
    gndr: str | None = '소녀'
    cstm: str | None = ''
    look: str | None = ''
    bgnd: str | None = ''
    sampler_name: str | None = 'DPM++ 2M'
    save_images: bool | None = False
    scheduler: str | None = 'karras'


def text2imgapi2(self, txt2imgreq: Txt2ImgParam):

    hangul_text_arr = [txt2imgreq.gndr, txt2imgreq.cstm, txt2imgreq.look, txt2imgreq.bgnd]

    # gender, costume, look, background
    translator = deepl.Translator(API_KEY_DEEPL, server_url=API_URL_DEEPL)
    trans_rslt = translator.translate_text(hangul_text_arr, source_lang="KO", target_lang="EN-US")
    gender_noun = 'he'
    if trans_rslt[0].text is 'girl':
        gender_noun = 'she'
    new_pstv_pmpt_prefix = pstv_pmpt_prefix.replace('{{gender}}', trans_rslt[0].text.replace(';', ', BREAK,'))
    new_pstv_pmpt_prefix = new_pstv_pmpt_prefix + ', ' + gender_noun + ' is wearing ' + trans_rslt[1].text.replace(';', ', BREAK,')
    new_pstv_pmpt_prefix = new_pstv_pmpt_prefix + ', ' + gender_noun + ' is ' + trans_rslt[2].text.replace(';', ', BREAK,')
    new_pstv_pmpt_prefix = new_pstv_pmpt_prefix + ', ' + gender_noun + ' is ' + trans_rslt[3].text.replace(';', ', BREAK,')
    new_pstv_pmpt_prefix += pstv_pmpt_suffix

    task_id = txt2imgreq.force_task_id or create_task_id("txt2img")

    script_runner = scripts.scripts_txt2img

    infotext_script_args = {}
    self.apply_infotext(txt2imgreq, "txt2img", script_runner=script_runner, mentioned_script_args=infotext_script_args)

    selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

    populate = txt2imgreq.copy(update={  # Override __init__ params
        "sampler_name": api.validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
        "do_not_save_samples": not txt2imgreq.save_images,
        "do_not_save_grid": not txt2imgreq.save_images,
    })
    if populate.sampler_name:
        populate.sampler_index = None  # prevent a warning later on

    args = vars(populate)
    args.pop('script_name', None)
    args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
    args.pop('alwayson_scripts', None)
    args.pop('infotext', None)

    script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts,
                                        selectable_script_idx, script_runner, input_script_args=infotext_script_args)

    send_images = args.pop('send_images', True)
    args.pop('save_images', None)

    add_task_to_queue(task_id)

    with self.queue_lock:
        with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
            p.is_api = True
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples

            try:
                shared.state.begin(job="scripts_txt2img")
                start_task(task_id)
                if selectable_scripts is not None:
                    p.script_args = script_args
                    processed = scripts.scripts_txt2img.run(p, *p.script_args)  # Need to pass args as list here
                else:
                    p.script_args = tuple(script_args)  # Need to pass args as tuple here
                    processed = process_images(p)
                finish_task(task_id)
            finally:
                shared.state.end()
                shared.total_tqdm.clear()

    b64images = list(map(api.encode_pil_to_base64, processed.images)) if send_images else []

    return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())
