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

API_URL_DEEPL = 'https://api-free.deepl.com/v2/translate'
API_KEY_DEEPL = 'DeepL-Auth-Key 9b534bc9-7729-44ae-b4f4-2fc0ba33968a:fx'
pgtn_pmpt_prefix = "masterpiece, best quality, (realistic, photo-realistic:1.4), (RAW photo:1.2), extremely detailed CG unity 8k wallpaper, an extremely delicate and beautiful, amazing, finely detail, official art, huge filesize, ultra-detailed, extremely detailed, beautiful detailed girl, extremely detailed eyes and face, beautiful detailed eyes, light on face, (((dramatic lighting, full body))), dramatic lighting, (masterpiece:1.2), (best_quality:1.4), (highres:1.1), (ultra-detailed:1.2), subsurface scattering, (sharp:0.7), amazing fine detail, Nikon D850 film stock photograph Kodak Portra 400 camera f1.6 lens, rich colors, lifelike texture, dramatic lighting, sidelighting, (Polaroid dark tone low key film:1.1),ultra-detailed, ultra high res, side looks, professional lighting, hyperrealism,(sidelighting, finely detailed beautiful eyes: 1.3), sexy, beautiful, big eyes, beautiful detailed eyes, finely detailed beautiful eyes, high quality makeup, cute, slim waist, aegyo sal, 1{{gender}}, (1 mid 20's a little muscular kpop idol {{gender}}:1.1), "
pgtv_pmpt_suffix = ", <lora:kitagawa_marin_v1-1:1>, <lora:Asian orgasm v4:1.3>"
ngtv_pmpt     = "(worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)),paintings, sketches,nipples, skin spots, acnes, skin blemishes, bad anatomy,facing away, looking away,tilted head,lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions, gross, easynegative, negative_hand-neg, wearing vests, ng_deepnegative_v1_75t, daytime, bright, 2girls, 2 girls, two girls, petals, logo, nametag, watermark, tattoo, tattoos, daytime, bright, leather, cum, dripping, (sweat), wet, tattoo, veins, FastNegativeV2, fat, v, peace symbol, reflected light in eyes"

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
    gndr: str | None = ''
    cstm: str | None = ''
    look: str | None = ''
    bgnd: str | None = ''


def text2imgapi2(self, txt2imgreq: Txt2ImgParam):

    hangul_text_arr =[txt2imgreq.gndr, txt2imgreq.cstm, txt2imgreq.look, txt2imgreq.bgnd]

    translator = deepl.Translator(API_KEY_DEEPL, server_url=API_URL_DEEPL)
    trans_rslt = translator.translate_text(hangul_text_arr, source_lang="KO", target_lang="EN")
    print(trans_rslt.text)

    return

    en_arr = trans_rslt

    pmpt = pgtn_pmpt_prefix + en_arr + pgtv_pmpt_suffix


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
