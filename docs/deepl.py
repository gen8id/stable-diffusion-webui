from fastapi import FastAPI
from pydantic import BaseModel
import deepl

API_URL_DEEPL    = 'https://api-free.deepl.com'
API_KEY_DEEPL    = '9b534bc9-7729-44ae-b4f4-2fc0ba33968a:fx'
pstv_pmpt_prefix = "(realistic, photo realistic, dramatic lighting, full body, best quality:1.4), (side lighting, finely detailed beautiful eyes: 1.3), (raw photo, masterpiece, ultra detailed:1.2), (highres, aegyo sal, Polaroid dark tone low key film:1.1), extremely detailed CG unity 8k wallpaper, extremely delicate and beautiful, amazing, finely detail, official art, huge filesize, extremely detailed, extremely detailed eyes and face, light on face, subsurface scattering, amazing fine detail, Nikon D850 film stock photograph Kodak Portra 400 camera f1.6 lens, rich colors, lifelike texture, ultra high res, side look, professional lighting, hyper realism, sexy, beautiful, big eyes, beautiful detailed eyes, high quality makeup, cute, slim waist, 1{{gender}}, (mid 20's a little muscular kpop idol {{gender}}:1.1), (sharp:0.7)"
pstv_pmpt_suffix = ", <lora:kitagawa_marin_v1-1:1>, <lora:Asian orgasm v4:1.3>"
ngtv_pmpt        = "(worst quality, low quality, normal quality:1.8), (monochrome, grayscale:1.2), (sweat:1.1), lowres, paintings, sketches, nipples, skin spots, acnes, skin blemishes, bad anatomy, tilted head, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross, easynegative, negative_hand-neg, wearing vests, ng_deepnegative_v1_75t, 2girls, 2 girls, two girls, petals, logo, nametag, watermark, tattoo, tattoos, leather, cum, dripping, wet, tattoo, veins, FastNegativeV2, fat, v, peace symbol, nsfw,  dirty face, 1leg, 1arm, 1 leg, 1 arm, one leg, one arm, blurred eyes, too big breast, naked, bare chested, underwear, lingerie, too short pants, naked abdomen, naked belly, naked stomach, 2head, 2 head, two head, reflected light in eyes"

app = FastAPI()


class Txt2ImgParam(BaseModel):
    gndr: str | None = ''
    cstm: str | None = ''
    look: str | None = ''
    bgnd: str | None = ''


@app.post("/api/v1/txt2img")
def text2imgapi2(txt2imgreq: Txt2ImgParam):

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

    return {"message": f"{new_pstv_pmpt_prefix}"}

